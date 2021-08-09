import sys
import os.path
import pprint
import numpy as np
from subprocess import call
import itertools
import argparse





def parseVals (fname):

    f = open(fname)
    line = f.readline()
    rtvals = []
    schemes = []
    while line:
        if (line == '' or line == '\n'):
            break
        tmp = line.split()
        if (tmp[0] == 'matrix'):
            schemes = tmp[1:]
            print 'number of schemes:', len(schemes)
            print 'scheme names:', ' '.join(schemes)
            line = f.readline()
            continue

        mat = tmp[0]
        tmp = tmp[1:]
        assert len(tmp) == len(schemes)
        for (rt,sch) in zip(tmp,schemes):
            rtvals.append((sch, float(rt), mat, 1.0/float(rt)))

        line = f.readline()

    f.close()

    return schemes, rtvals



def perfProfVals (schemes, rtvals, args):
    '''

    partitioning instance = (mat)

    '''

    inst = {}
    for x in rtvals:
        t = x[2]
        if (t not in inst):
            inst[t] = []
        inst[t].append((x[0], x[3]))

    # normalize
    instNew = {}
    for k,v in inst.iteritems():
        best = min([x[1] for x in v])
        instNew[k] = []
        for i in range(len(v)):
            x = list(v[i])
            x[1] = float(x[1])/float(best)
            instNew[k].append(tuple(x))

    inst = instNew
    ninst = len(inst)
    sys.stdout.write('Number of instances: %d\n' % (ninst))
    # pp.pprint(inst)

    xmax = max([x[3] for x in rtvals])
    xticks = [round(x,3) for x in np.arange(1.00, xmax, args.xstep)]
    # xticks = [round(x,3) for x in np.arange(1.00, args.xmax, args.xstep)]
    # xticks.extend(np.linspace(args.xmax, xmax, 10000))
    perfvals = {}
    for sch in schemes:
        perfvals[sch] = [0 for x in range(len(xticks))]

    for k,v in inst.iteritems():
        for x in v:
            sch = x[0]
            normval = x[1]
            for i in range(len(xticks)):
                if (normval <= xticks[i]):
                    perfvals[sch][i] += 1

    # percentages
    for k,v in perfvals.iteritems():
        for i in range(len(v)):
            v[i] = round(float(v[i])/float(ninst), 3)

    return perfvals, xticks


def genPlot (schemes, perfvals, xticks, title_str, args):

    outdatafile = open((args.result_file + '-perf-prof.dat'), 'w')
    outgpfile = open((args.result_file + '-perf-prof.p'), 'w')
    s = '# within%\t'
    for sch in schemes:
        s += str(sch) + '\t'
    s += '\n'
    outdatafile.write(s)

    yticks = [round(y,3) for y in np.arange(0.05, 1.05, args.ystep)]
    yticksVals = []
    for y_idx in range(len(yticks)):
        y = yticks[y_idx]
        yticksVals.append([])
        for sch in schemes:
            vals = perfvals[sch]
            x = '?'
            for i in range(len(xticks)):
                if (y <= vals[i]):
                    x = xticks[i]
                    break
            yticksVals[y_idx].append(x) # one value per scheme
        print yticksVals[y_idx]

    for x_idx in range(len(xticks)):
        x = xticks[x_idx]
        for y_idx in range(len(yticks)):
            y = yticks[y_idx]
            s = str(x) + '\t'
            for schVal in yticksVals[y_idx]:
                if (str(schVal) == '?'):
                    s += '?\t'
                elif (schVal == x):
                    s += str(y) + '\t'
                else:
                    # sys.stdout.write('this should not happen.\n')
                    s += '?\t'
            s += '\n'
            outdatafile.write(s)
    outdatafile.close()

    outgpfile.write('set term postscript eps enhanced color\n')
    outgpfile.write('set output \'' +
                    args.result_file + '-perf-prof.eps' +
                    '\'\n')
    outgpfile.write("set size 1.00,1.50\n")
    outgpfile.write("unset log\n")
    outgpfile.write("unset label\n")
    outgpfile.write("set xrange [%f:%f]\n" %
                    (0.98, args.xmax))
    outgpfile.write("set yrange [0:1.0]\n")
    xstr = "set xtics ("
    for x in np.arange(1.00, args.xmax, 0.10):
        xstr += "\"" + str(round(x,2)) + "\"" + " " + str(round(x,2)) + ", "
    xstr = xstr[0:len(xstr)-2]
    xstr += ") font \", 18\""
    outgpfile.write("%s\n" % (xstr))
    ystr = "set ytics ("
    for y in np.arange(0.00, 1.10, 0.10):
        ystr += "\"" + str(round(y,2)) + "\"" + " " + str(round(y,2)) + ", "
    ystr = ystr[0:len(ystr)-2]
    ystr += ") font \", 18\""
    outgpfile.write("%s\n" % (ystr))

    outgpfile.write('set title \"' +
                    title_str +
                    '\" font \", 24\"\n')
    outgpfile.write('set xlabel \"Parallel runtime relative to the best'
                    '\" font \", 24\"\n')
    outgpfile.write('set ylabel \"fraction of test cases\" font \", 24\"\n')
    outgpfile.write('set bmargin 14\n')

    # outgpfile.write("set key inside right bottom spacing 1.40 font \", 20\"\n")
    # outgpfile.write("set key horizontal outside center bottom spacing 1.40 font \", 24\"\n")

    # pointIds = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 5, 7] # count = 12
    # pointIds = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # count = 12
    pointIds = [4, 6, 8, 10, 12, 14, 1, 2]
    pointSizes = [1.35, 1.60, 1.90, 1.90, 1.80, 1.70, 1.80, 1.80]
    colorIds = ['#DC143C', '#FF7F50', '#DA70D6', '#3CB371', '#808080', '#0000FF', '#B8860B', '#008B8B']
    # colorIds = ['#000000'] * len(schemes)
    lineTypes = [1, 1, 1, 1,
                 1, 1, 1, 1,
                 1, 1, 1, 1]

    outgpfile.write("set key horizontal outside center bottom\n")

    colorIds = ['#DC143C', '#FF7F50', '#DA70D6', '#3CB371', '#808080', '#0000FF']
    pointIds = [-1, 0, 1, 2, 3, -1, 4, 6, 8, 10, 12, 4, 5, 7, 9, 11, 13, 5]


    oldLen = len(colorIds)
    while len(colorIds) < len(schemes):
        colorIds.append(colorIds[len(colorIds) % oldLen])

    oldLen = len(lineTypes)
    while len(lineTypes) < len(schemes):
        lineTypes.append(lineTypes[len(lineTypes) % oldLen])

    oldLen = len(pointIds)
    while len(pointIds) < len(schemes):
        pointIds.append(pointIds[len(pointIds) % oldLen])

    oldLen = len(pointSizes)
    while len(pointSizes) < len(schemes):
        pointSizes.append(pointSizes[len(pointSizes) % oldLen])

    for i in range(len(schemes)):
        outgpfile.write('set style line %d lc rgb \'%s\' '
                        'lt %d lw 2 pt %d ps %.2f\n' %
                        (i+1, colorIds[i], lineTypes[i],
                         pointIds[i], pointSizes[i]))

    outgpfile.write('plot ')
    for i in range(len(schemes)):
        if (i < len(schemes)-1):
            outgpfile.write('\"' +
                            args.result_file + '-perf-prof.dat' +
                            '\" u 1:%d t \'%s\' '
                            'w linespoints ls %d, \\\n\t' %
                            (i+2, '%s' % (schemes[i]), i+1))
        else:
            outgpfile.write('\"' +
                            args.result_file + '-perf-prof.dat' +
                            '\" u 1:%d t \'%s\' '
                            'w linespoints ls %d\n' %
                            (i+2, '%s' % (schemes[i]), i+1))

    outgpfile.write("set output\n")
    outgpfile.close()

    call(["gnuplot", args.result_file + '-perf-prof.p'])

    return



parser = argparse.ArgumentParser()
parser.add_argument('result_file')
parser.add_argument('--xstep', type=float, default=0.010)
parser.add_argument('--xmax', type=float, default=5)
parser.add_argument('--ystep', type=float, default=0.025)

args = parser.parse_args()

schemes, rtvals = parseVals(args.result_file)
perfvals, xticks = perfProfVals(schemes, rtvals, args)
genPlot(schemes, perfvals, xticks, "All-phase schemes except heap (cori-KNL)", args)
# genPlot(schemes, perfvals, xticks, "Two-phase schemes (cori-KNL)", args)
# genPlot(schemes, perfvals, xticks, "One-phase schemes (cori-KNL)", args)
