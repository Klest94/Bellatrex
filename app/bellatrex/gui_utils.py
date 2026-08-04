def check_and_import_gui_dependencies():
    """Import and return the NiceGUI backend.

    Raises:
        ImportError: If NiceGUI is not installed.
    """
    try:
        import nicegui
    except ImportError as exc:
        raise ImportError(
            "The NiceGUI dependency is not installed. Install it with: pip install bellatrex"
        ) from exc

    return nicegui
