
def check_and_import_gui_dependencies():
    try:
        import dearpygui
        import dearpygui_ext
    except ImportError as e:
        raise ImportError(
            "Optional dependencies for the GUI are not installed, "
            "Namely dearpygui>=1.6.2 and dearpygui-ext>=0.9.5. "
            "Please install them using: pip install bellatrex[gui]"
        ) from e
    return dearpygui, dearpygui_ext