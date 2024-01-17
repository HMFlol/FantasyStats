import gspread

gc = gspread.oauth()

sh = gc.open("FantasyStats")

print(sh.sheet1.get("A1"))
