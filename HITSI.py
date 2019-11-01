## @package HITSI
## Takes a large list of command line arguments
## and collects the files in dictionaries,
## gets their SVD,
## and then plots all the results.
from plot_attributes import *
from psitet_load import loadshot
from utilities import SVD, \
     plot_chronos, plot_itor
import click

@click.command()
@click.option('--directory', \
    default='/media/removable/SD Card/Two-Temperature-Post-Processing/', \
    help='Directory containing the .mat files')
@click.option('--filenames', \
    default=['exppsi_129499.mat'],multiple=True, \
    help='A list of all the filenames, which '+ \
        'allows a large number of shots to be '+ \
        'compared')
@click.option('--freqs', \
    default=14.5,multiple=True, \
    help='A list of all the injector frequencies (kHz) which '+ \
        'correspond to the list of filenames')
@click.option('--limits', \
    default=(0.0,1.0),type=(float,float),multiple=True, \
    help='Time limits for each of the discharges')
@click.option('--trunc', \
    default=10,type=int, \
    help='Where to truncate the SVD')

## Main program that accepts python 'click' command line arguments.
## Note that options with multiple=true must have multiple values added
## separately. This format could also be done
## by declaring --limits to be of type (int,int). If a description
## of the various click options is desired, just type
## python HITSI.py --help
def analysis(directory,filenames,freqs,limits,trunc):

    print('Running with the following command line options: ')
    print('Truncation number for the SVD = ', trunc)
    print('Directory where the files to analyze reside: ',directory)
    print('File(s) to load = ',filenames)
    print('Frequencies corresponding to those files = ',freqs)
    print('Time limits for each of the files = ',limits)

    is_HITSI3 = False
    if(len(filenames[0])==9):
        is_HITSI3=True
    filenames=np.atleast_1d(filenames)
    freqs=np.atleast_1d(freqs)
    total = []
    for i in range(len(filenames)):
        filename = filenames[i]
        f_1 = np.atleast_1d(freqs[i])
        if filenames[i][0:10]=='Psi-Tet-2T':
            temp_dict = loadshot('Psi-Tet-2T',directory, \
                int(f_1),True,True,is_HITSI3,limits[i])
        elif filenames[i][0:3]=='Psi':
            temp_dict = loadshot('Psi-Tet',directory, \
                int(f_1),True,False,is_HITSI3,limits[i])
        else:
            temp_dict = loadshot(filename,directory, \
                np.atleast_1d(int(f_1)),False,False, \
                is_HITSI3,limits[i])
        temp_dict['use_IMP'] = False
        temp_dict['trunc'] = trunc
        temp_dict['f_1'] = f_1
        total.append(temp_dict)

    total = np.asarray(total).flatten()
    color_ind = 0
    for i in range(len(filenames)):
        if i % 4 == 0 and i != 0:
            color_ind = color_ind + 1
        color = colors2T[color_ind]
        SVD(total[i])
        plot_itor(total[i],(i%4)+1,color)
        plot_chronos(total[i],(i%4)+1,color)

if __name__ == '__main__':
    analysis()
