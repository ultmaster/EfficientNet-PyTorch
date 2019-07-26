import os
from argparse import ArgumentParser

import xlsxwriter


def analyze_metrics(root_dir, worksheet):
    # Some data we want to write to the worksheet.
    expenses = (
        ['Rent', 1000],
        ['Gas', 100],
        ['Food', 300],
        ['Gym', 50],
    )

    # Start from the first cell. Rows and columns are zero indexed.
    row = 0
    col = 0

    # Iterate over the data and write it out row by row.
    for item, cost in (expenses):
        worksheet.write(row, col, item)
        worksheet.write(row, col + 1, cost)
        row += 1

    # Write a total using a formula.
    worksheet.write(row, 0, 'Total')
    worksheet.write(row, 1, '=SUM(B1:B4)')



def main(root_dir):
    workbook = xlsxwriter.Workbook(os.path.join(root_dir, 'result.xlsx'))

    metrics = workbook.add_worksheet("metrics")

    workbook.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dir", help="Where the downloader stores the data in the last step")
    args = parser.parse_args()
    main(args.dir)
