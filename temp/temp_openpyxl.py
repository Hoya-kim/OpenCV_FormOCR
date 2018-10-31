# standard width openpyxl=2.117 cm/10.71 chars/ 80 pixels
# standard height openpyxl =0.529 cm/15 points/20 pixels
from openpyxl import Workbook

wb = Workbook()

ws = wb.active

final_x = [52, 174, 223, 355, 470, 586, 642, 741, 843, 931, 1017, 1237]
final_y = [4, 105, 158, 214, 265, 375, 441, 494, 548, 602, 656, 710, 773, 875, 933, 1722, 1776, 1813]

for y in range(0, len(final_y) - 2):
    ws.row_dimensions[y].height = final_y[y + 1] - final_y[y]
for x in range(0, len(final_x) - 2):
    ws.column_dimensions[x].width = final_x[x + 1] - final_x[x]

wb.save('test_openpyxl.xlsx')
