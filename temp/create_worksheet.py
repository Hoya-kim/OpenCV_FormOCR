# worksheet.set_landscape() --> 페이지 방향 가로
# worksheet.set_portrait() --> 페이지 방향 세로
# worksheet.set_paper(9) --> index 9 : A4

import xlsxwriter
import cv2


# 파라미터 final_x, final_y
def create_worksheet():
    '''
    workbook = xlsxwriter.Workbook('./data/result/test.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet1 = workbook.add_worksheet()
    worksheet2 = workbook.add_worksheet()

    worksheet.set_row(0, 20)  # Set the height of Row 1 to 20.

    cell_format = workbook.add_format({'bold': True})

    worksheet.set_row(0, 20, cell_format)

    worksheet.set_row(0, None, cell_format)  # Row 1 has cell_format

    worksheet.set_column(0, 0, 20)  # Column  A   width set to 20.
    worksheet.set_column(1, 3, 30)  # Columns B-D width set to 30.
    worksheet.set_column('E:E', 20)  # Column  E   width set to 20.
    worksheet.set_column('F:H', 30)  # Columns F-H width set to 30.

    worksheet.activate()

    merge_format = workbook.add_format({'align': 'center'})
    worksheet.merge_range(2, 1, 3, 3, 'Merged Cells', merge_format)
    worksheet.merge_range('B3:D4', 'Merged Cells', merge_format)
    '''

    # Create an new Excel file and add a worksheet.
    workbook = xlsxwriter.Workbook('test.xlsx')
    worksheet = workbook.add_worksheet()

    data_format1 = workbook.add_format({
        'font_name' : 'Malgun Gothic',
        'align': 'center',
        'border': 1,
        'border_color': 'red'
        })  # border_color

    # Create a format to use in the merged range.
    merge_format = workbook.add_format({
        'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'fg_color': 'yellow'})

    # Merge 3 cells.
    # worksheet.merge_range('B4:D4', 'Merged Range', merge_format)

    # Merge 3 cells over two rows.
    # worksheet.merge_range('B7:D8', 'Merged Range', merge_format)

    final_x = [52, 174, 223, 355, 470, 586, 642, 741, 843, 931, 1017, 1237]
    final_y = [4, 105, 158, 214, 265, 375, 441, 494, 548, 602, 656, 710, 773, 875, 933, 1722, 1776, 1813]

    for y in range(0, len(final_y) - 2):
        worksheet.set_row(y, int(final_y[y + 1] - final_y[y]), data_format1)  # height

    for x in range(0, len(final_x) - 2):
        worksheet.set_column(x, x, int(final_x[x + 1] - final_x[x]) / 7.5, data_format1)  # width #/7

    worksheet.write('A1', 'ddd',data_format1)
    workbook.close()
    # worksheet = workbook.add_format()


create_worksheet()
