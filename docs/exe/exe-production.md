# How To Create An Executable File Using Auto PY To EXE

There are steps to create an executable file using Auto PY To EXE. These steps are tested on Windows 10 using one of `pixi` environment called `exe-build`, which is available in `sdb_gui` repository. These are the steps and you are allowed to experiment with the settings to find the best one for you.

1. Run `auto-py-to-exe` in terminal and a graphic interface will show.
2. Insert `sdb_gui.py` in script location.
3. Use One File and Console Based.
4. Insert `sdb_gui.ico` in icon.
5. Insert the following additional file and directories in Additional Files:
   1. `LICENSE` and `pixi.toml` files --add files.
   2. config, gui, icons, licenses, and sdb directories --add folder.
   3. GDAL directory (usually in \your-directory\Library\share\gdal) --add folder.
6. Go to Advanced options,
   1. add hidden import:
      1. rasterio.sample
      2. pyogrio._geometry
      3. fiona
   2. add runtime hook and select `hook-gdal.py`.
   3. add exclude module.
      1. PySide6
      2. torch
7. Go to Settings options and specify your output directory.
8. CONVERT .PY TO .EXE
