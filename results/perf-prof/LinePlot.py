import argparse
from subprocess import call


def parse_input(file_name, series):
    schemas = []
    x_values = []
    y_values = []

    with open(file_name) as f:
        for i, line in enumerate(f):
            tokens = line.rstrip().split()
            if i == 0:
                for v in tokens:
                    if v == 'matrix':
                        continue
                    schemas.append(v)
            else:
                x_values.append(int(tokens[0]))
                y_values.append(tokens[1:])

    return schemas, x_values, y_values


def gnu_plot(schemas, x_values, y_values, args):
    with open(args.output + '.dat', 'w') as f:
        f.write("#")
        for schema in schemas:
            f.write(schema + '\t')
        f.write('\n')

        assert len(x_values) == len(y_values)

        for row in range(len(y_values)):
            f.write('%d\t' % x_values[row])
            for y in y_values[row]:
                f.write(y + '\t')
            f.write('\n')

    with open(args.output + '.gpi', 'w') as f:
        f.write('set term postscript eps enhanced color\n')
        f.write('set output \'' + args.output + '.eps' + '\'\n')
        f.write("set size 1.00,1.5\n")
        f.write("unset log\n")
        f.write("unset label\n")
        f.write("set xrange [%f:%f]\n" % (min(x_values), max(x_values)))

        f.write('set xtics(' +
                ', '.join(['"' + str(round(x, 2)) + '"' + " " + str(round(x, 2)) for x in x_values])
                + ') font ", 18"\n')
        # Set yrange,  x and y ticks

        f.write('set title \"' + args.title + '\" font \", 24\"\n')
        f.write('set xlabel \"' + args.xlabel + '\" font \", 24\"\n')
        f.write('set ylabel \"' + args.ylabel + '\" font \", 24\"\n')

        if args.style == 1:
            point_ids = [5, 7, 13, 1, 2, 3]
            point_sizes = [1.5] * 6
            line_types = [1] * 6
            color_ids = ['#DC143C', '#0000FF', '#FF7F50', '#DA70D6', '#3CB371', '#808080']
            f.write("set key horizontal opaque inside left top maxcolumns 2 font \",24\n")
        elif args.style == 2:
            point_ids = [5, 7, 1, 2, 3]
            point_sizes = [1.5] * 5
            line_types = [1] * 5
            color_ids = ['#DC143C', '#0000FF', '#DA70D6', '#3CB371', '#808080']
            f.write("set key horizontal opaque inside left top maxcolumns 2 font \",24\n")
        else:
            point_ids = [i for i in range(0, 18)]
            point_sizes = [1.5] * 18
            line_types = [1] * 18
            color_ids = ['#DC143C', '#FF7F50', '#DA70D6', '#3CB371', '#808080', '#0000FF'] * 3
            f.write('set bmargin 8\n')
            f.write("set key horizontal outside center bottom maxcolumns 2 font \",18\n")

        # set styles
        for i, s in enumerate(range(len(args.series.split(',')))):
            f.write('set style line %d lc rgb \'%s\' lt %d lw 2 pt %d ps %.2f\n' %
                    (i + 1, color_ids[i], line_types[i], point_ids[i], point_sizes[i]))

        f.write('plot ')
        first = True
        for i, v in enumerate([int(i) for i in args.series.split(',')]):
            if first:
                first = False
            else:
                f.write(', \\\n\t')

            f.write('"' + args.output + '.dat\" u 1:%d t \'%s\' w linespoints ls %d' %
                    (v + 2, '%s' % (schemas[v]), i + 1))
        f.write('\n')
        f.write("set output\n")

    call(["gnuplot", args.output + '.gpi'])


def latex_plot(schemas, x_values, y_values, args):

    series = [int(i) for i in args.series.split(',')]
    entries = [schemas[i] for i in series]

    if args.style == 1:
        colors = ['#DC143C', '#0000FF', '#FF7F50', '#DA70D6', '#3CB371']
        marks = ['square*', '*', 'diamond*', '|', 'x']
        legend_columns = 3
    elif args.style == 2:
        colors = ['#DC143C', '#0000FF', '#DA70D6', '#3CB371']
        marks = ['square*', '*', '|', 'x']
        legend_columns = 2
    else:
        legend_columns = 1
        colors = []
        marks = []

    colors = ['rgb,255:red,%d;green,%d;blue,%d' % (c / 256 / 256, c / 256 % 256, c % 256 % 256) for c in
              [int(c.replace('#', ''), 16) for c in colors]]

    with open(args.output + '.tex', 'w') as f:
        if args.standalone:
            f.write('\\documentclass{standalone}\n'
                    '\\usepackage{pgfplots}\n'
                    '\\usepackage{tikz}\n'
                    '\\usepgfplotslibrary{groupplots}\n'
                    '\\pgfplotsset{compat=1.16}\n'
                    '\\begin{document}\n')

        f.write('\\begin{centering}\n'
                '\\scalebox{1.0} {\n'
                '\\begin{tikzpicture}\n')

        # Group plot
        f.write('    \\begin{groupplot}[\n'
                '        legend columns=%d,\n' % legend_columns +
                '        legend entries={%s},\n' % (', '.join(entries)) +
                '        legend cell align={left},\n'
                '        legend to name={%s},\n' % args.output +
                '        legend style={draw=none},\n'
                '        y label style={at={(axis description cs:0.05,.5)}},\n'
                '        group style={\n'
                '                group size=1 by 1,\n'
                '                xlabels at=edge bottom,\n'
                '                ylabels at=edge left,\n'
                '                horizontal sep=1.5cm,\n'
                '                vertical sep=1.5cm\n'
                '                },\n'
                '        xlabel = {%s},\n' % args.xlabel +
                '        ylabel = {%s},\n' % args.ylabel +
                '        xmin=%d,\n' % min(x_values) +
                '        xmax=%d,\n' % max(x_values) +
                # '        ymin=,\n'
                # '        ymax=,\n'
                '        xtick={%s},\n' % (', '.join([str(x) for x in x_values])) +
                # '        ytick={,\n'
                '        ylabel near ticks,\n'
                '        yticklabel style={\n'
                '            /pgf/number format/precision=3,\n'
                '            /pgf/number format/fixed\n'
                '        },\n'
                '        scaled y ticks=false,\n'
                '    ]\n')

        f.write('\n')
        f.write('    \\nextgroupplot[title={' + args.title + '}]\n')

        for i, s in enumerate(series):
            f.write('    \\addplot+[color={%s}, mark=%s, mark options={fill={%s}}]\n' % (colors[i], marks[i], colors[i]))
            f.write('    coordinates { %s };\n'
                    % ' '.join(['(' + str(x) + ', ' + y_values[j][s] + ')' for j, x in enumerate(x_values)]))

        f.write('    \\end{groupplot}\n')
        f.write('\\end{tikzpicture}\n'
                '}\n')
        f.write('\\pgfplotslegendfromname{%s}\n' % args.output)
        f.write('\\end{centering}\n')

        if args.standalone:
            f.write('\\end{document}\n')

    if args.standalone:
        call(["pdflatex", args.output + '.tex'])
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--series', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--title', type=str, default="Title")
    parser.add_argument('--xlabel', type=str, default="X axis")
    parser.add_argument('--ylabel', type=str, default="Y axis")
    parser.add_argument('--style', type=int, default=0)
    parser.add_argument('--standalone', type=bool, default=False)

    args = parser.parse_args()

    schemas, x_values, y_values = parse_input(args.input, args.series)

    gnu_plot(schemas, x_values, y_values, args)
    # latex_plot(schemas, x_values, y_values, args)

if __name__ == '__main__':
    main()
