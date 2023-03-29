import itertools
from game_stats import GameStats
import keyboard
import cv2
import time
import numpy as np
from dataclasses import dataclass
import parse

from logger import Logger
from config import Config
import template_finder
from utils.misc import wait, is_in_roi, mask_by_roi
from utils.custom_mouse import mouse
from inventory import stash, common, vendor
from ui import view
from ui_manager import detect_screen_object, is_visible, select_screen_object_match, wait_until_visible, ScreenObjects, center_mouse, wait_for_update
from messages import Messenger
from d2r_image import processing as d2r_image
from d2r_image.data_models import HoveredItem, ItemTooltip
from screen import grab, convert_screen_to_monitor
from item import consumables
from bnip.NTIPAliasStat import NTIPAliasStat as NTIP_STATS
from bnip.actions import should_id, should_keep

inv_gold_full = False
messenger = Messenger()

nontradable_items = ["key of ", "essense of", "wirt's", "jade figurine"]

@dataclass
class InventoryItemAnalyze:
    img: np.ndarray = None # Tooltip image
    pos: tuple = None # Slot coordinates on screen
    column: int = None # Column index in inventory
    row: int = None # Row index in inventory
    need_id: bool = False
    sell: bool = False
    keep: bool = False
    def __getitem__(self, key):
        return super().__getattribute__(key)
    def __setitem__(self, key, value):
        setattr(self, key, value)

def get_inventory_gold_full():
    return inv_gold_full

def set_inventory_gold_full(new_value: bool):
    global inv_gold_full
    if get_inventory_gold_full() != new_value:
        Logger.info(f"Set inventory gold full: {new_value}")
        inv_gold_full = new_value

def inventory_has_items(close_window = False) -> bool:
    """
    Check if Inventory has any items
    :return: Bool if inventory still has items or not
    """
    open()
    img = grab()
    items=False
    for column, row in itertools.product(range(0, Config().char["num_loot_columns"]), range(4)):
        _, slot_img = common.get_slot_pos_and_img(img, column, row)
        if common.slot_has_item(slot_img):
            items=True
            break
    if close_window:
        common.close()
    if items:
        return True
    return False


def stash_all_items(items: list = None):
    """
    Stashing all items in inventory. Stash UI must be open when calling the function.
    """
    global messenger
    if not get_inventory_gold_full() and items is None:
        Logger.debug("No items to stash, skip")
        return None
    center_mouse()
    # Wait for stash to fully load
    if not common.wait_for_left_inventory():
        Logger.error("stash_all_items(): Failed to find stash menu. Continue...")
        return items
    # stash gold
    if Config().char["stash_gold"]:
        if not is_visible(ScreenObjects.GoldNone):
            Logger.debug("Stashing gold")
            common.select_tab(min(3, stash.get_curr_stash()["gold"]))
            wait(0.7, 1)
            stash_full_of_gold = False
            # Try to read gold count with OCR
            try: stash_full_of_gold = common.read_gold(grab(), "stash") == 2500000
            except: pass
            if not stash_full_of_gold:
                # If gold read by OCR fails, fallback to old method
                gold_btn = detect_screen_object(ScreenObjects.GoldBtnInventory)
                select_screen_object_match(gold_btn)
                # move cursor away from button to interfere with screen grab
                mouse.move(-60, 0, absolute=False, randomize=15, delay_factor=[0.1, 0.3])
                if wait_until_visible(ScreenObjects.DepositBtn, 3).valid:
                    keyboard.send("enter") #if stash already full of gold just nothing happens -> gold stays on char -> no popup window
                else:
                    Logger.error("stash_all_items(): deposit button not detected, failed to stash gold")
                # if 0 gold becomes visible in personal inventory then the stash tab still has room for gold
                stash_full_of_gold = not wait_until_visible(ScreenObjects.GoldNone, 2).valid
            if stash_full_of_gold:
                Logger.debug("Stash tab is full of gold, selecting next stash tab.")
                stash.set_curr_stash(gold = (stash.get_curr_stash()["gold"] + 1))
                if Config().general["info_screenshots"]:
                    cv2.imwrite("./log/screenshots/info/info_gold_stash_full_" + time.strftime("%Y%m%d_%H%M%S") + ".png", grab())
                if stash.get_curr_stash()["gold"] > 3:
                    #decide if gold pickup should be disabled or gambling is active
                    vendor.set_gamble_status(True)
                else:
                    # move to next stash tab
                    return stash_all_items(items=items)
            else:
                set_inventory_gold_full(False)
    if not items:
        return []
    # check if stash tab is completely full (no empty slots)
    common.select_tab(stash.get_curr_stash()["items"])
    while stash.get_curr_stash()["items"] <= 3:
        img = grab()
        if is_visible(ScreenObjects.EmptyStashSlot, img):
            break
        else:
            Logger.debug(f"Stash tab completely full, advance to next")
            if Config().general["info_screenshots"]:
                cv2.imwrite("./log/screenshots/info/stash_tab_completely_full_" + time.strftime("%Y%m%d_%H%M%S") + ".png", img)
            if Config().char["fill_shared_stash_first"]:
                stash.set_curr_stash(items = (stash.get_curr_stash()["items"] - 1))
            else:
                stash.set_curr_stash(items = (stash.get_curr_stash()["items"] + 1))
            if (Config().char["fill_shared_stash_first"] and stash.get_curr_stash()["items"] < 0) or stash.get_curr_stash()["items"] > 3:
                stash.stash_full()
            common.select_tab(stash.get_curr_stash()["items"])
    # stash stuff
    while True:
        items = transfer_items(items, "stash")
        if items and any([item.keep for item in items]):
            # could not stash all items, stash tab is likely full
            Logger.debug("Wanted to stash item, but it's still in inventory. Assumes full stash. Move to next.")
            if Config().general["info_screenshots"]:
                cv2.imwrite("./log/screenshots/info/debug_info_inventory_not_empty_" + time.strftime("%Y%m%d_%H%M%S") + ".png", grab())
            if Config().char["fill_shared_stash_first"]:
                stash.set_curr_stash(items = (stash.get_curr_stash()["items"] - 1))
            else:
                stash.set_curr_stash(items = (stash.get_curr_stash()["items"] + 1))
            if (Config().char["fill_shared_stash_first"] and stash.get_curr_stash()["items"] < 0) or stash.get_curr_stash()["items"] > 3:
                stash.stash_full()
            common.select_tab(stash.get_curr_stash()["items"])
        else:
            break
    Logger.debug("Done stashing")
    return items

def open() -> np.ndarray:
    img = grab()
    if not common.inventory_is_open():
        keyboard.send(Config().char["inventory_screen"])
        if not wait_until_visible(ScreenObjects.RightPanel, 1).valid:
            if not view.return_to_play():
                return False
            keyboard.send(Config().char["inventory_screen"])
            if not wait_until_visible(ScreenObjects.RightPanel, 1).valid:
                Logger.error(f"personal.open(): Failed to open inventory")
                return False
    return True


def log_item(item_tooltip: ItemTooltip, hovered_item: HoveredItem):
    if item_tooltip is not None and item_tooltip.ocr_result:
        Logger.debug(f"OCR mean confidence: {item_tooltip.ocr_result.mean_confidence}")
        Logger.debug(hovered_item.Text)
        if Config().general["info_screenshots"]:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            found_low_confidence = False
            for cnt, x in enumerate(item_tooltip.ocr_result.word_confidences):
                if x <= 88:
                    try:
                        Logger.debug(f"Low confidence word #{cnt}: {item_tooltip.ocr_result.original_text.split()[cnt]} -> {item_tooltip.ocr_result.text.split()[cnt]}, Conf: {x}, save screenshot")
                        found_low_confidence = True
                    except: pass
            if found_low_confidence:
                cv2.imwrite(f"./log/screenshots/info/ocr_low_confidence_box_{timestamp}.png", item_tooltip.img)

def log_item_fail(hovered_item_screen, slot):
    Logger.error(f"item segmentation failed for slot_pos: {slot[0]}")
    if Config().general["info_screenshots"]:
        cv2.imwrite("./log/screenshots/info/failed_item_box_" + time.strftime("%Y%m%d_%H%M%S") + ".png", hovered_item_screen)

def inspect_inventory_items(close_window: bool = True, game_stats: GameStats = None, ignore_sell: bool = False) -> list[InventoryItemAnalyze]:
    """
    Iterate over all picked items in inventory--ID items and decide which to stash
    """
    
    analyzes = []
    quality = 'ticklish'
    
    center_mouse() # Place the mouse outside of the inventory
    open() # Open inventory if not already done
    
    inventory_screen = grab() # Screenshot of the inventory without any item hovered/tooltip shown
    
    # If called during gambling or buying/selling, 
    vendor_open = is_visible(ScreenObjects.GoldBtnVendor, inventory_screen)

    # List occupied slots
    occupied_slots = []
    for column, row in itertools.product(range(Config().char["num_loot_columns"]), range(4)):
        slot_pos, slot_img = common.get_slot_pos_and_img(inventory_screen, column, row)
        if common.slot_has_item(slot_img):
            occupied_slots.append([slot_pos, row, column])
    
    # We will avoid treating a same slot many times
    already_detected_item_tooltip_rois = [] # Already treated regions of the inventory
    already_treated_slots = set() # Already treated slots (row, col)
    
    # Iterate over occupied inventory slots
    for count, slot in enumerate(occupied_slots):
        failed = False
        
        # Ignore this slot if it lies within the range of a previous item's dimension property
        if (slot[1], slot[2]) in already_treated_slots:
            continue
        # Ignore this slot if it lies within in a previous item's ROI (no dimension property)
        if any(is_in_roi(item_roi, slot[0]) for item_roi in already_detected_item_tooltip_rois):
            continue
        
        delay = [0.2, 0.3] if count else [1, 1.3] # Longer mouse move delay for the 1st item for realism
        
        # Move the mouse over the inventory slot
        x_m, y_m = convert_screen_to_monitor(slot[0])
        mouse.move(x_m, y_m, randomize = 10, delay_factor = delay)
        wait(0.1, 0.2)
        hovered_item_screen = grab(True)
        
        # Retrieve the item properties & dimensions
        hovered_item, item_tooltip = d2r_image.get_hovered_item_data(hovered_item_screen)
        
        if hovered_item and item_tooltip and item_tooltip.ocr_result:
            # An item has been detected
            
            # Register already treated slots (if possible), otherwise already treated ROIs
            if hasattr(hovered_item, 'BaseItem') and "dimensions" in hovered_item.BaseItem:
                # The item dimension is known
                # By construction "slot" is the top-left slot of the item, so together with the dimension we can compute the set of all slots treated with this item
                positions = common.dimensions_to_slots(hovered_item.BaseItem["dimensions"], (slot[1], slot[2]))
                already_treated_slots.update(positions)
            else:
                Logger.warning(f"Unknown item dimension for slot {slot}, marking ROI as treated")
                already_detected_item_tooltip_rois.append(item_tooltip.roi)
            

            # determine whether the item can be sold
            tooltip_text = item_tooltip.ocr_result.text.splitlines()
            item_name = vendor_open and tooltip_text[1] or tooltip_text[0]
            item_can_be_traded = not any(substring in item_name for substring in nontradable_items) # quest items are not tradable

            # Format the output
            analyze = InventoryItemAnalyze(
                img = item_tooltip.img,
                pos = (x_m, y_m),
                column = slot[2],
                row = slot[1],
                sell = Config().char["sell_junk"] and not any(substring in item_name for substring in nontradable_items) and not ignore_sell,
                need_id = is_visible(ScreenObjects.Unidentified, item_tooltip.img) and should_id(hovered_item.as_dict()),
                keep = False
            )
            
            # First, identify items that need identification
            if (analyze.need_id):
                # Move the mouse out to remove the item tooltip
                center_mouse()
                
                # Try grabbing an ID tome
                tome_state, tome_pos = common.tome_state(grab(True), tome_type = "id", roi = Config().ui_roi["restricted_inventory_area"])
                if tome_state is not None and tome_state == "ok":
                    # Identify the item with it
                    common.id_item_with_tome([x_m, y_m], tome_pos)
                    
                    # recapture analyze after ID
                    mouse.move(x_m, y_m, randomize = 4, delay_factor = delay)
                    wait(0.05, 0.1)
                    hovered_item_screen = grab(True)
                    
                    # Check identification result
                    hovered_item, item_tooltip = d2r_image.get_hovered_item_data(hovered_item_screen)
                    if hovered_item and item_tooltip and item_tooltip.ocr_result:
                        identification_succeed = not is_visible(ScreenObjects.Unidentified, item_tooltip.img)
                        if identification_succeed:
                            # Note: analyze.sell doesn't change when identifying
                            analyze.img = item_tooltip.img
                            analyze.need_id = False
                        else:
                            Logger.warning(f"Failed to identify item in slot {slot}")
                    else:
                        # An error occurred in image cropping or parsing
                        failed = True
                else:
                    Logger.warning(f"ID Tome not found or empty when identifying item in slot {slot}")

            # Choose what to do of the item
            if not failed:
                log_item(item_tooltip, hovered_item)
                
                analyze.keep, expression = should_keep(hovered_item.as_dict())

                # make sure it's not a consumable
                # TODO: logic for trying to add potion to belt if there are needs
                analyze.keep &= not bool(consumables.is_consumable(hovered_item))

                if analyze.keep:
                    # Item to be kept
                    Logger.info(f"Keep {item_name}. Expression: {expression}")
                    analyzes.append(analyze)
                    if game_stats is not None:
                        game_stats.log_item_keep(item_name, True, item_tooltip.img, item_tooltip.ocr_result.text, expression, hovered_item.as_dict())
                elif analyze.need_id:
                    # Item to be identified later
                    Logger.debug(f"Need to ID {item_name}.")
                    analyzes.append(analyze)
                elif analyze.sell:
                    if vendor_open:
                        # Sell now
                        Logger.debug(f"Selling {item_name}.")
                        transfer_items([analyze], action = "sell")
                    else:
                        # Keep and sell later
                        Logger.debug(f"Need to sell {item_name}.")
                        analyzes.append(analyze)
                else:
                    #Logger.debug(f"Discarding {json.dumps(hovered_item.as_dict(), indent = 4)}")
                    Logger.debug(f"Discarding {item_name}.")
                    transfer_items([analyze], action = "drop")
                wait(0.05, 0.2)
            else:
                failed = True
        else: # get_hovered_item_data did not end properly
            failed = True
        
        if failed:
            log_item_fail(hovered_item_screen, slot)

    if close_window:
        common.close()
    return analyzes

def transfer_items(items: list, action: str = "drop", img: np.ndarray = None) -> list:
    #requires open inventory / stash / vendor
    if not items:
        return []
    img = img if img is not None else grab(True)
    filtered = []
    left_panel_open = is_visible(ScreenObjects.LeftPanel, img)
    match action:
        case "drop":
            filtered = [ item for item in items if item.keep == False and item.sell == False ]
        case "sell":
            filtered = [ item for item in items if item.keep == False and item.sell == True ]
            if not left_panel_open:
                Logger.error(f"transfer_items: Can't perform, vendor is not open")
        case "stash":
            if is_visible(ScreenObjects.GoldBtnStash, img):
                filtered = [ item for item in items if item.keep == True ]
            else:
                Logger.error(f"transfer_items: Can't perform, stash is not open")
        case _:
            Logger.error(f"transfer_items: incorrect action param={action}")
    if filtered:
        # if dropping, control+click to drop unless left panel is open, then drag to middle
        # if stashing, control+click to stash
        # if selling, control+click to sell
        if (action == "drop" and not left_panel_open) or action in ["sell", "stash"]:
            keyboard.send('ctrl', do_release=False)
            wait(0.1, 0.2)
        for item in filtered:
            pre_hover_img = grab(True)
            _, slot_img = common.get_slot_pos_and_img(pre_hover_img, item.column, item.row)
            if not common.slot_has_item(slot_img):
                # item no longer exists in that position...
                Logger.debug(f"Item at {item.pos} doesn't exist, skip and remove from list")
                for cnt, o_item in enumerate(items):
                    if o_item.pos == item.pos:
                        items.pop(cnt)
                        break
                continue
            # move to item position and left click
            mouse.move(*item.pos, randomize=4, delay_factor=[0.2, 0.4])
            wait(0.2, 0.4)
            pre_transfer_img = grab(True)
            mouse.press(button="left")
            # wait for inventory image to update indicating successful transfer / item select
            success = wait_for_update(pre_transfer_img, Config().ui_roi["open_inventory_area"], 3)
            mouse.release(button="left")
            if not success:
                Logger.warning(f"transfer_items: inventory unchanged after attempting to {action} item at position {item.pos}")
                break
            else:
                # if dropping, drag item to middle if vendor/stash is open
                if action == "drop" and left_panel_open:
                    center_mouse()
                    wait(0.04, 0.08)
                    mouse.press(button="left")
                    wait(0.2, 0.3)
                    mouse.release(button="left")
                # item successfully transferred, delete from list
                Logger.debug(f"Confirmed {action} at position {item.pos}")
                for cnt, o_item in enumerate(items):
                    if o_item.pos == item.pos:
                        items.pop(cnt)
                        break
                if action == "sell":
                    # check and see if inventory gold count changed
                    if (gold_unchanged := not wait_for_update(pre_transfer_img, Config().ui_roi["inventory_gold_digits"], 3)):
                        Logger.info("Inventory gold is full, force stash")
                    set_inventory_gold_full(gold_unchanged)
    keyboard.send('ctrl', do_press=False)
    return items

def update_tome_key_needs(img: np.ndarray = None, item_type: str = "tp") -> bool:
    open()
    img = grab()
    if item_type.lower() in ["tp", "id"]:
        match = template_finder.search(
            [f"{item_type.upper()}_TOME", f"{item_type.upper()}_TOME_RED"],
            img,
            roi = Config().ui_roi["restricted_inventory_area"],
            best_match = True,
            )
        if match.valid:
            if match.name == f"{item_type.upper()}_TOME_RED":
                consumables.set_needs(item_type, 20)
                return True
            # else the tome exists and is not empty, continue
        else:
            Logger.debug(f"update_tome_key_needs: could not find {item_type}")
            return False
    elif item_type.lower() in ["key"]:
        match = template_finder.search("INV_KEY", img, roi = Config().ui_roi["restricted_inventory_area"])
        if not match.valid:
            return False
    else:
        Logger.error(f"update_tome_key_needs failed, item_type: {item_type} not supported")
        return False
    mouse.move(*match.center_monitor, randomize=4, delay_factor=[0.5, 0.7])
    wait(0.2, 0.2)
    hovered_item_screen = grab(True)
    # get the item description box
    hovered_item, item_tooltip = d2r_image.get_hovered_item_data(hovered_item_screen)
    if item_tooltip is not None:
        try:
            quantity = int(hovered_item.NTIPAliasStat[NTIP_STATS["quantity"]])
            max_quantity = int(hovered_item.NTIPAliasStat[NTIP_STATS["quantitymax"]])
            consumables.set_needs(item_type, max_quantity - quantity)
        except Exception as e:
            Logger.error(f"update_tome_key_needs: unable to parse quantity for {item_type}. Exception: {e}")
    else:
        Logger.error(f"update_tome_key_needs: Failed to capture item description box for {item_type}")
        if Config().general["info_screenshots"]:
            cv2.imwrite("./log/screenshots/info/failed_capture_item_description_box" + time.strftime("%Y%m%d_%H%M%S") + ".png", hovered_item_screen)
        return False
    return True