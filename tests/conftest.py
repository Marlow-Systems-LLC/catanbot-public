import sys
import types

import pytest

# Off macOS (the Docker image): stand-in pyobjc modules so catanbot.mac imports and
# the tests that monkeypatch it run. Tests marked needs_macos (real OCR or capture) are skipped.
if sys.platform != "darwin":
    from unittest import mock

    def _stub_attr(attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return mock.MagicMock(name=attr)

    for _name in ("Quartz", "AppKit", "Vision", "Foundation", "ApplicationServices"):
        _m = types.ModuleType(_name)
        _m.__getattr__ = _stub_attr
        sys.modules.setdefault(_name, _m)


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_macos: uses Apple Vision OCR or screen capture")


# tests that read live frames with Apple Vision OCR or call the macOS layer for real
NEEDS_MACOS = {
    "test_board_anchor.py": {"test_scan_lattice_passed_in_is_anchored_then_fixed"},
    "test_game_over_exit.py": None, "test_leave_flow.py": {"test_live_quit_dialogs_are_read_from_the_frame"},
    "test_live_fixes_20261006.py": {"test_hand_reader_keeps_short_crane_title",
                                    "test_improve_dialog_ok_found_when_ocr_misses_it",
                                    "test_leftover_improve_dialog_is_cancelled",
                                    "test_setup_first_harbour_then_numbers_last"},
    "test_monopoly_pick.py": None, "test_popup_ocr.py": None,
    "test_probe_controls.py": {"test_bank_ratios_read_from_the_dialog", "test_defender_deck_prompt_picks_a_deck",
                               "test_deserter_player_dialog_clicks_the_check_under_the_name",
                               "test_two_digit_count_and_the_slots_after_it", "test_enlarged_card_is_not_a_player_pick",
                               "test_spy_pick_clicks_a_card_under_its_title", "test_prompt_text_letters_are_not_rings",
                               "test_bank_dialog_shows_our_real_counts", "test_five_progress_titles_read_for_the_discard",
                               "test_science_bonus_pick_clicks_the_top_row_and_the_check"},
    "test_review_031.py": {"test_reanswered_offer_is_booked_once"},
    "test_runner.py": {"test_no_rules_no_play", "test_setup_after_rules", "test_roll_then_end_or_trade",
                       "test_declined_trade_closes", "test_incoming_offer_declined",
                       "test_third_setup_settlement_is_not_a_city_loss", "test_setup_road_takes_the_ringed_edge"},
    "test_stage_1008.py": {"test_rookie_fixture_reads_three_seats", "test_remove_city_fixture_classified"},
    "test_stage_1009.py": {"test_harbor_fixture_classified"},
}


def pytest_collection_modifyitems(config, items):
    if sys.platform == "darwin":
        return
    skip = pytest.mark.skip(reason="needs macOS (Apple Vision / Quartz)")
    for item in items:
        names = NEEDS_MACOS.get(item.path.name, set())
        if "needs_macos" in item.keywords or names is None or item.originalname in names:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _private_game_flags(tmp_path, monkeypatch):
    """Tests must never write the live bot's data/game_flags.json (a test once left its
    Greater Catan rules there, which a restarted live runner would have loaded)."""
    from catanbot.runner import Runner
    monkeypatch.setattr(Runner, "FLAGS", tmp_path / "game_flags.json")


@pytest.fixture(autouse=True)
def _private_shots(tmp_path, monkeypatch):
    """Tests must never prune or index the live bot's screenshots (catanbot/shots.py)."""
    from catanbot import shots
    monkeypatch.setattr(shots, "DATA", tmp_path / "shots_data")
    monkeypatch.setattr(shots, "INDEX", tmp_path / "shots_data" / "shots.json")
    monkeypatch.setattr(shots, "_cache", None)


@pytest.fixture(autouse=True)
def _private_stop_file(tmp_path, monkeypatch):
    """The server keeps data/STOP while it only trains; the runner tests must not see it."""
    from catanbot import runner
    monkeypatch.setattr(runner, "STOP", tmp_path / "STOP")


@pytest.fixture(autouse=True)
def _default_setup_weights(monkeypatch):
    """Tests check the code's setup defaults, not whatever scripts/fit_setup.py accepted into
    data/config.json "setup_weights" (tests that need a config patch catanbot.config.load).
    "solo" is per machine too (on for the server's training box): tests start from a live game."""
    import catanbot.config as cfg
    real = cfg.load
    monkeypatch.setattr(cfg, "load", lambda: {k: v for k, v in real().items() if k not in ("setup_weights", "solo")})


@pytest.fixture(autouse=True)
def _private_game_log(tmp_path, monkeypatch):
    """Runner tests must never write or archive the live bot's data/game_logs."""
    from catanbot import game_log
    monkeypatch.setattr(game_log, "DIR", tmp_path / "game_logs")


@pytest.fixture(autouse=True)
def _private_record(tmp_path, monkeypatch):
    """Tests must never count games in the live bot's data/record.json."""
    from catanbot import chat
    monkeypatch.setattr(chat, "RECORD_PATH", tmp_path / "record.json")
