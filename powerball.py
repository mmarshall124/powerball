import argparse
import time
import numpy as np
import sys

from src.diversity import shannon_diversity, lottery_shannon
from src.groups import group_submatrices, filter_groups
from src.null import null_dists_comp, null_dists_lottery
from src.csv_IO import csv_reader, csv_writer
from src.dataMatrix import OutputMatrix
from src.statFunctions import p_values, standard_effect, raw_effect
from src.graphics import bokeh_xy, bokeh_xy_cmap, bokeh_xy_sliders


def main():
    startTime = time.time()
    args = arg_parser()

    # Reading input from file
    filename = args.filename
    if args.verbose:
        print("Processing input file.")
    inputMatrix = csv_reader(filename, args.noHeaders)

    if args.verbose:
        print("Generating group submatrices.")
    # TODO Rename inputMatrix.uniqueGroups to something more representative of what it's now used for.
    # Returns a filtered list of unique groups that are above the min size
    inputMatrix.uniqueGroups = filter_groups(inputMatrix.groups, args.groupSize)

    # Array of the species within a group across samples for each group. 3D
    groupSubmatrices = group_submatrices(inputMatrix.data, inputMatrix.groups,
                                         inputMatrix.uniqueGroups)
    if args.verbose:
        print("Generating null distributions.")
    # Array of null distributions for each group. Used to calculate
    # competitiveness. 3D
    nullDistsComp = null_dists_comp(inputMatrix.data, inputMatrix.groups,
                                    args.nullSize)

    # Array of null distributions for each group. Used to calculate
    # Lottery scores. 3D
    nullDistsLot = null_dists_lottery(inputMatrix.data, inputMatrix.groups,
                                      args.nullSize)

    if args.verbose:
        print("Generating diversity scores.")
    # List of empirical shannon diversity data for each group. 2D
    compDiversities = [shannon_diversity(group) for group in groupSubmatrices]

    # List of empirical shannon diversity data from the row sums of each group
    # 2D
    lotDiversities = [lottery_shannon(group) for group in groupSubmatrices]

    # Collecting output data into an object
    outputMatrix = output(inputMatrix, nullDistsComp, nullDistsLot,
                          compDiversities, lotDiversities, startTime, args)

    # Writing to file
    csv_writer(args.outputFile, outputMatrix.outputArray)

    # Graphing
    if args.chartStyle == 1:
        bokeh_xy(outputMatrix)
    elif args.chartStyle == 2:  # Default
        bokeh_xy_cmap(outputMatrix)
    elif args.chartStyle == 3:
        bokeh_xy_sliders(outputMatrix)
    # Values above 4 silence chart output


def output(inputMatrix, nullDistsComp, nullDistsLot, compDiversities,
           lotDiversities, startTime,
           args):
    # Collects all output data and prints to console (arg dependent)

    if args.verbose:  # Verbose printing
        print("\nVerbose output:")
        print("data:\n", inputMatrix.data)
        print("groups:", inputMatrix.groups)
        print("Number of groups:", len(inputMatrix.uniqueGroups))
        print("Sample Headers:", inputMatrix.sampleHeaders)
        print("Species Headers:", inputMatrix.speciesHeaders)
        print("")

    # Empty arrays to fill during the main loop
    compScores = np.empty(len(inputMatrix.uniqueGroups))
    compPValues = np.empty(len(inputMatrix.uniqueGroups))
    lotScores = np.empty(len(inputMatrix.uniqueGroups))
    lotPValues = np.empty(len(inputMatrix.uniqueGroups))

    # Storing and printing all output data
    for i in range(len(inputMatrix.uniqueGroups)):

        # By default uses raw effect size, configurable with arguments.
        # Multiplied by -1 to give more user-friendly, positive values.
        if args.standardEffect:
            compScores[i] = standard_effect(nullDistsComp[i],
                                            compDiversities[i]) * -1
        else:
            compScores[i] = raw_effect(nullDistsComp[i],
                                            compDiversities[i]) * -1

        compPValues[i] = p_values(nullDistsComp[i], compDiversities[i],
                                  args.twoTailed, args.alpha)

        if args.standardEffect:
            lotScores[i] = standard_effect(nullDistsLot[i], lotDiversities[i]) * -1
        else:
            lotScores[i] = raw_effect(nullDistsLot[i], lotDiversities[i]) * -1

        # Lottery P-Value is always one-tailed
        lotPValues[i] = p_values(nullDistsLot[i], lotDiversities[i], False,
                                 args.alpha)

        if not args.quiet:  # Printing all data to console
            print(inputMatrix.uniqueGroups[i], "Competitiveness:",
            "{0:.4f}".format(compScores[i]), end=" ")
            print("p-value:", "{0:.4f}".format(compPValues[i]))

            print(inputMatrix.uniqueGroups[i], "Lottery score:",
            "{0:.4f}".format(lotScores[i]), end=" ")
            print("p-value:", "{0:.4f}".format(lotPValues[i]))

            print()  # newline

    if not args.quiet:
        print("--- %s seconds ---" % (time.time() - startTime))  # Runtime

    # Collecting all the data into an array
    outputArray = np.transpose(np.array([inputMatrix.uniqueGroups,
                                         compScores, compPValues,
                                         lotScores, lotPValues]))

    # Creating an array with all output data
    outputMatrix = OutputMatrix(lotScores, compScores, lotPValues, compPValues,
                                inputMatrix.uniqueGroups,
                                inputMatrix.sampleHeaders,
                                inputMatrix.speciesHeaders,
                                outputArray)

    return outputMatrix


def arg_parser():
    # Handles command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("filename",
                        help="Location of the csv file to read as input.")
    parser.add_argument("-nh", "--noHeaders",
                        help=("Specifies that the input file doesn't have"
                              " headers."),
                        action="store_true")
    parser.add_argument("-tt", "--twoTailed",
                        help=("Uses a two-tailed p-value instead of one-tailed"
                              ". When calculating competitiveness p-value"),
                        action="store_true")
    parser.add_argument("-ns", "--nullSize",
                        help=("Determines the sample size of null "
                              "distributions (Default = 1000)"),
                        type=int, default=1000)
    parser.add_argument("-o", "--outputFile",
                        help=("Specifies the output csv filename "
                              "(Default = Output.csv)"),
                        type=str, default="Output.csv")
    parser.add_argument("-gs", "--groupSize",
                        help=("Determines the minimum group size"
                              "(Default = 3)"),
                        type=int, default=3)
    parser.add_argument("-a", "--alpha",
                        help=("Determines the alpha for measuring P-values"
                              "(Default = 0.05)"),
                        type=float, default=0.05)
    parser.add_argument("-se", "--standardEffect",
                        help=("Uses standard effect size for computing competitiveness"
                        " and lottery scores instead of the difference."),
                        action="store_true")
    parser.add_argument("-cs", "--chartStyle",
                        help=("Determines the type of chart to display (1-3)"
                              "1. Scatterplot where significance of both "
                              "values are shown through size."
                              "2. Scatterplot where significance of comp. is"
                              " shown through size, and significance of lot "
                              "is shown through color. (Default)"
                              "3. Scatterplot where significance is filtered "
                              "using sliders. Higher values silence chart"
                              " output."),
                        type=int, default=2)

    group = parser.add_mutually_exclusive_group()  # Quiet vs Verbose
    group.add_argument("-q", "--quiet",
                       help="Disables printing to console",
                       action="store_true")
    group.add_argument("-v", "--verbose",
                       help="Prints all read data from the input file.",
                       action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    main()
