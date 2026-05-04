"""Application entry point."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication

from app.config.settings import load
from app.db.schema import init_db
from app.ui.app_context import AppContext
from app.ui.main_window import MainWindow, load_harmony_sans


def _rebuild_providers(ctx: AppContext, compare, qa) -> None:
    """Reload provider and embedder from current settings, then refresh pages."""
    from app.core.model.factory import get_embedder, get_provider

    try:
        ctx.provider = get_provider(ctx.settings)
    except Exception:
        ctx.provider = None
    try:
        ctx.embedder = get_embedder(ctx.settings)
    except Exception:
        ctx.embedder = None

    compare.refresh_versions()
    qa.refresh_documents()
    qa.refresh_compare_tasks()


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("DocDiffAgent")
    _ico_path = Path(__file__).parent / "assets" / "icons" / "docdiff.ico"
    if _ico_path.exists():
        app.setWindowIcon(QIcon(str(_ico_path)))

    harmony_family = load_harmony_sans()
    if harmony_family:
        app.setFont(QFont(harmony_family, 13))

    settings = load()

    data_dir = (
        Path(settings.data_dir)
        if settings.data_dir
        else Path.home() / "AppData" / "Local" / "DocDiffAgent" / "data"
    )
    data_dir.mkdir(parents=True, exist_ok=True)

    conn = init_db(str(data_dir))

    from app.core.model.factory import get_embedder, get_provider

    try:
        provider = get_provider(settings)
    except Exception:
        provider = None
    try:
        embedder = get_embedder(settings)
    except Exception:
        embedder = None

    ctx = AppContext(
        settings=settings,
        conn=conn,
        data_dir=str(data_dir),
        provider=provider,
        embedder=embedder,
    )

    window = MainWindow(ctx)

    from app.ui.pages.compare_page import ComparePage
    from app.ui.pages.home_page import HomePage
    from app.ui.pages.library_page import LibraryPage
    from app.ui.pages.qa_page import QaPage
    from app.ui.pages.settings_page import SettingsDialog

    home = HomePage(ctx)
    compare = ComparePage(ctx)
    library = LibraryPage(ctx)
    qa = QaPage(ctx)
    settings_dialog = SettingsDialog(ctx, parent=window)

    window.add_page(0, home)
    window.add_page(1, compare)
    window.add_page(2, library)
    window.add_page(3, qa)

    home.navigate_requested.connect(window.navigate_to)
    window.settings_requested.connect(settings_dialog.exec)
    settings_dialog.provider_changed.connect(
        lambda: _rebuild_providers(ctx, compare, qa)
    )

    window.navigate_to(0)

    home.refresh()
    library.refresh()
    compare.refresh_versions()
    qa.refresh_documents()
    qa.refresh_compare_tasks()

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
