import argparse
import sys
from subprocess import call


def parse_input(file_name):
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
    label_upper_left_haswell = 'set label "Haswell" at graph  0.03, graph 0.9 font ",32" textcolor rgb "#FF0000"'
    label_upper_left_knl = 'set label "KNL" at graph  0.03, graph 0.9 font ",32" textcolor rgb "#FF0000"'

    label_upper_right_haswell = 'set label "Haswell" at graph  0.72, graph 0.9 font ",32" textcolor rgb "#FF0000"'
    label_upper_right_knl = 'set label "KNL" at graph  0.89, graph 0.9 font ",32" textcolor rgb "#FF0000"'

    label_bottom_right_haswell = 'set label "Haswell" at graph  0.8, graph 0.1 font ",32" textcolor rgb "#FF0000"'
    label_bottom_right_knl = 'set label "KNL" at graph  0.89, graph 0.1 font ",32" textcolor rgb "#FF0000"'

    margin = ''
    label = ''
    scale = ''
    if args.name == 'tricnt-rmat-haswell-scaling-flops':
        title = "Triangle counting - scaling - Haswell"
        xlabel = "Number of Threads"
        ylabel = "GFLOPS"
        size_x = 1.0
        size_y = 0.5
        series = [3, 2, 4, 1, 0]
        point_ids = [5, 7, 13, 1, 2, 3]
        colors = ['#DC143C', '#0000FF', '#FF7F50', '#DA70D6', '#3CB371']
        legend = 'set key horizontal outside right center maxcolumns 1 font ",24'
        label = label_upper_left_haswell
        # scale = 'set logscale xy'
    elif args.name == 'tricnt-rmat-haswell-scaling-teps':
        title = "Triangle counting - scaling - Haswell"
        xlabel = "Number of Threads"
        ylabel = "MTEPS"
        size_x = 1.0
        size_y = 0.5
        series = [3, 2, 4, 1, 0]
        point_ids = [5, 7, 13, 1, 2, 3]
        colors = ['#DC143C', '#0000FF', '#FF7F50', '#DA70D6', '#3CB371']
        legend = 'set key horizontal outside right center maxcolumns 1 font ",24'
        # scale = 'set logscale xy'
    elif args.name == 'tricnt-rmat-knl-scaling-flops':
        title = "Triangle counting - scaling - KNL"
        xlabel = "Number of Threads"
        ylabel = "GFLOPS"
        size_x = 1.0
        size_y = 0.5
        series = [3, 2, 4, 1, 0]
        point_ids = [5, 7, 13, 1, 2, 3]
        colors = ['#DC143C', '#0000FF', '#FF7F50', '#DA70D6', '#3CB371']
        legend = 'set key horizontal outside right center maxcolumns 1 font ",24'
        label = label_upper_left_knl
        # scale = 'set logscale xy'
    elif args.name == 'tricnt-rmat-knl-scaling-teps':
        title = "Triangle counting - scaling - KNL"
        xlabel = "Number of Threads"
        ylabel = "TEPS"
        size_x = 1.0
        size_y = 0.5
        series = [3, 2, 4, 1, 0]
        point_ids = [5, 7, 13, 1, 2, 3]
        colors = ['#DC143C', '#0000FF', '#FF7F50', '#DA70D6', '#3CB371']
        legend = 'set key horizontal outside right center maxcolumns 1 font ",24'
        # scale = 'set logscale xy'
    elif args.name == 'tricnt-rmat-haswell-flops':
        title = "Triangle counting - Haswell"
        xlabel = "Scale"
        ylabel = "GFLOPS"
        size_x = 1.0
        size_y = 0.5
        series = [9, 8, 10, 3, 2]
        point_ids = [5, 7, 13, 1, 2, 3]
        colors = ['#DC143C', '#0000FF', '#FF7F50', '#DA70D6', '#3CB371']
        legend = 'set key horizontal outside center right maxcolumns 1 font ",24'
        label = label_upper_left_haswell
    elif args.name == 'tricnt-rmat-haswell-teps':
        title = "Triangle counting - Haswell"
        xlabel = "Scale"
        ylabel = "MTEPS"
        size_x = 1.0
        size_y = 0.5
        series = [9, 8, 10, 3, 2]
        point_ids = [5, 7, 13, 1, 2, 3]
        colors = ['#DC143C', '#0000FF', '#FF7F50', '#DA70D6', '#3CB371']
        legend = 'set key horizontal outside center right maxcolumns 1 font ",24'
        label = label_upper_left_haswell
    elif args.name == 'tricnt-rmat-knl-flops':
        title = "Triangle counting - KNL"
        xlabel = "Scale"
        ylabel = "GFLOPS"
        size_x = 1.0
        size_y = 0.5
        series = [9, 8, 10, 3, 2]
        point_ids = [5, 7, 13, 1, 2, 3]
        colors = ['#DC143C', '#0000FF', '#FF7F50', '#DA70D6', '#3CB371']
        legend = 'set key horizontal outside center right maxcolumns 1 font ",24'
        label = label_upper_left_knl
    elif args.name == 'tricnt-rmat-knl-teps':
        title = "Triangle counting - KNL"
        xlabel = "Scale"
        ylabel = "TEPS"
        size_x = 1.0
        size_y = 0.5
        series = [9, 8, 10, 3, 2]
        point_ids = [5, 7, 13, 1, 2, 3]
        colors = ['#DC143C', '#0000FF', '#FF7F50', '#DA70D6', '#3CB371']
        legend = 'set key horizontal outside center right maxcolumns 1 font ",24'
        label = label_upper_left_knl
    elif args.name == 'ktruss-5-rmat-haswell-flops':
        title = "K-Truss - Haswell"
        xlabel = "Scale"
        ylabel = "GFLOPS"
        size_x = 1.0
        size_y = 0.5
        series = [8, 7, 9, 10, 2, 1]
        point_ids = [5, 7, 13, 15, 1, 2, 3]
        colors = ['#DC143C', '#0000FF', '#FF7F50', '#808080', '#DA70D6', '#3CB371']
        legend = 'set key horizontal outside center right maxcolumns 1 font ",24'
        label = label_upper_left_haswell
    elif args.name == 'ktruss-5-rmat-haswell-teps':
        title = "K-Truss - Haswell"
        xlabel = "Scale"
        ylabel = "MTEPS"
        size_x = 1.0
        size_y = 0.5
        series = [8, 7, 9, 10, 2, 1]
        point_ids = [5, 7, 13, 15, 1, 2, 3]
        colors = ['#DC143C', '#0000FF', '#FF7F50', '#808080', '#DA70D6', '#3CB371']
        legend = 'set key horizontal outside center right maxcolumns 1 font ",24'
        label = label_upper_right_haswell
    elif args.name == 'ktruss-5-rmat-knl-flops':
        title = "K-Truss - KNL"
        xlabel = "Scale"
        ylabel = "GFLOPS"
        size_x = 1.0
        size_y = 0.5
        series = [8, 7, 9, 10, 2, 1]
        point_ids = [5, 7, 13, 15, 1, 2, 3]
        colors = ['#DC143C', '#0000FF', '#FF7F50', '#808080', '#DA70D6', '#3CB371']
        legend = 'set key horizontal outside center right maxcolumns 1 font ",24'
        label = label_upper_left_knl
    elif args.name == 'ktruss-5-rmat-knl-teps':
        title = "K-Truss - KNL"
        xlabel = "Scale"
        ylabel = "MTEPS"
        size_x = 1.0
        size_y = 0.5
        series = [8, 7, 9, 10, 2, 1]
        point_ids = [5, 7, 13, 15, 1, 2, 3]
        colors = ['#DC143C', '#0000FF', '#FF7F50', '#808080', '#DA70D6', '#3CB371']
        legend = 'set key horizontal outside center right maxcolumns 1 font ",24'
        label = label_upper_left_knl
    elif args.name == 'bc-rmat-haswell-512':
        title = "Betweenness centrality - Haswell"
        xlabel = "Scale"
        ylabel = "TEPS"
        size_x = 1.0
        size_y = 0.5
        series = [6, 5, 2, 1]
        point_ids = [5, 7, 1, 2, 3]
        colors = ['#DC143C', '#0000FF', '#DA70D6', '#3CB371']
        legend = 'set key horizontal outside right center maxcolumns 1 font ",24'
        label = label_upper_left_haswell
    elif args.name == 'bc-rmat-knl-512':
        title = "Betweenness centrality - KNL"
        xlabel = "Scale"
        ylabel = "TEPS"
        size_x = 1.0
        size_y = 0.5
        series = [6, 5, 2, 1]
        point_ids = [5, 7, 1, 2, 3]
        colors = ['#DC143C', '#0000FF', '#DA70D6', '#3CB371']
        legend = 'set key horizontal outside right center maxcolumns 1 font ",24'
        label = label_upper_left_knl
    else:
        title = "Title"
        xlabel = "xlabel"
        ylabel = "ylable"
        size_x = 1.0
        size_y = 0.5
        series = [1] * len(schemas)
        point_ids = [1] * len(schemas)
        colors =["#000000"] * len(schemas)
        legend = ''
        margin = ''
        sys.stderr.write('name not supported')

    point_sizes = [1.5] * len(schemas)
    line_types = [1] * len(schemas)

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

    # Generate gpi file
    with open(args.output + '.gpi', 'w') as f:
        f.write('set term postscript eps enhanced color\n')
        f.write('set output \'' + args.output + '.eps' + '\'\n')
        f.write("set size %f,%f\n" % (size_x, size_y))
        f.write("unset log\n")
        if label:
            f.write(label + '\n')

        f.write("set xrange [%f:%f]\n" % (min(x_values), max(x_values)))

        if x_values.count(68) == 0:
            f.write('set xtics(' +
                    ', '.join(['"' + str(round(x, 2)) + '"' + " " + str(round(x, 2)) for x in x_values])
                    + ') font ", 18"\n')
        else:
            f.write('set xtics(' +
                    ', '.join(['"' + (str(round(x, 2)) if x != 2 else '') + '"' + " " + str(round(x, 2))
                               for x in x_values])
                    + ') font ", 18"\n')
        # Set yrange,  x and y ticks

        # if title:
        #     f.write('set title \"' + title + '\" font \", 24\"\n')
        f.write('set xlabel \"' + xlabel + '\" font \", 24\"\n')
        f.write('set ylabel \"' + ylabel + '\" font \", 24\"\n')

        if scale:
            f.write(scale + '\n')

        if margin:
            f.write(margin)

        if legend:
            f.write(legend + '\n')

        # set styles
        for i in range(len(series)):
            f.write('set style line %d lc rgb \'%s\' lt %d lw 2 pt %d ps %.2f\n' %
                    (i + 1, colors[i], line_types[i], point_ids[i], point_sizes[i]))

        f.write('set multiplot\n')
        f.write("set size %f,%f\n" % (size_x, size_y))
        f.write('plot ')
        first = True
        for i, v in enumerate([int(i) for i in series]):
            if first:
                first = False
            else:
                f.write(', \\\n\t')

            if v >= 0:
                f.write('"' + args.output + '.dat\" u 1:%d t \'%s\' w linespoints ls %d' %
                        (v + 2, '%s' % (schemas[v]), i + 1))
            else:
                f.write('NaN lt -2 ti " "\n')

        f.write('\n')
        f.write('unset multiplot\n')

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
            f.write(
                '    \\addplot+[color={%s}, mark=%s, mark options={fill={%s}}]\n' % (colors[i], marks[i], colors[i]))
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
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--standalone', type=bool, default=False)

    args = parser.parse_args()

    if not args.output:
        args.output = args.input.replace('.txt', '')

    if not args.name:
        args.name = args.input.replace('.txt', '')

    schemas, x_values, y_values = parse_input(args.input)

    gnu_plot(schemas, x_values, y_values, args)
    # latex_plot(schemas, x_values, y_values, args)


if __name__ == '__main__':
    main()
