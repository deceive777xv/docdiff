def test_main_window_importable():
    from app.ui.main_window import MainWindow, NavButton, SideBar
    from app.ui.app_context import AppContext
    assert MainWindow is not None

def test_app_context_fields():
    import sqlite3
    from app.config.settings import AppSettings
    from app.ui.app_context import AppContext
    settings = AppSettings()
    conn = sqlite3.connect(":memory:")
    ctx = AppContext(settings=settings, conn=conn, data_dir="/tmp/test")
    assert ctx.provider is None
    assert ctx.embedder is None
    conn.close()

def test_settings_dialog_importable():
    from app.ui.pages.settings_page import SettingsDialog
    assert SettingsDialog is not None

def test_home_page_importable():
    from app.ui.pages.home_page import HomePage
    assert HomePage is not None

def test_library_page_importable():
    from app.ui.pages.library_page import LibraryPage
    assert LibraryPage is not None
