## @package HITSI
## Takes a large list of command line arguments
## and collects the files in dictionaries,
## gets their SVD,
## and then plots all the results.
from plot_attributes import *
from psitet_load import loadshot
from utilities import SVD, \
     plot_chronos, plot_itor, \
     plot_temperatures, plot_nFIR, \
     plot_centroid, plot_power_balance, \
     plot_individual_heat_flows, plot_navg, \
     plot_pressure, toroidal_modes_sp, plot_powers, \
     plot_all_heat_flows
import click
import os

@click.command()
@click.option('--directory', \
    default='/media/removable/SD Card/Two-Temperature-Post-Processing/', \
    help='Directory containing the .mat files')
@click.option('--filenames', \
    default=['expPSI_129499.mat'],multiple=True, \
    help='A list of all the filenames, which '+ \
        'allows a large number of shots to be '+ \
        'compared')
@click.option('--freqs', \
    default=14.5,multiple=True, \
    help='A list of all the injector frequencies (kHz) which '+ \
        'correspond to the list of filenames')
@click.option('--limits', \
    default=(0.0,1.0),type=(float,float), \
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

    filenames=np.atleast_1d(filenames)
    freqs=np.atleast_1d(freqs)
    total = []
    for j in range(len(filenames)):
        filename = filenames[j]
        is_HITSI3 = False
        if filename[7:9]=='-3':
            is_HITSI3=True
        for i in range(len(freqs)):
            f_1 = np.atleast_1d(freqs[i])
            print(i,j,filename,f_1,is_HITSI3)
            if '2T' in filename and 'PSI' in filename:
                temp_dict = loadshot(filename.rsplit('2T', 1)[0]+'2T',directory, \
                    int(f_1),True,True,is_HITSI3,limits)
            elif filename[0:3]=='PSI':
                temp_dict = loadshot('PSI-Tet',directory, \
                    int(f_1),True,False,is_HITSI3,limits)
            else:
                temp_dict = loadshot(filename+str(int(f_1))+'.mat',directory, \
                    np.atleast_1d(int(f_1)),False,False, \
                    is_HITSI3,limits)
            temp_dict['use_IMP'] = False
            temp_dict['trunc'] = trunc
            temp_dict['f_1'] = f_1
            total.append(temp_dict)

    total = np.asarray(total).flatten()
    color_ind = 0
    T2_ind = 0
    for i in range(len(freqs)*len(filenames)):
        SVD(total[i])
        if len(freqs) == 4:
            subpl1 = 2
            subpl2 = 2
        else:
            subpl1 = 1
            subpl2 = len(freqs)
        subpl3 = (i%(len(freqs)))+1
        if subpl3-1 == 0 and i != 0:
            color_ind = color_ind + 1
        color = colors2T[color_ind]
        plt.figure(75000,figsize=(figx, figy))
        plt.subplot(subpl1,subpl2,subpl3)
        plt.figure(85000,figsize=(figx, figy))
        plt.subplot(subpl1,subpl2,subpl3)
        plt.figure(95000,figsize=(figx, figy))
        plt.subplot(subpl1,subpl2,subpl3)
        plt.figure(105000,figsize=(figx, figy))
        plt.subplot(subpl1,subpl2,subpl3)
        plt.figure(115000,figsize=(figx, figy))
        plt.subplot(subpl1,subpl2,subpl3)
        plt.figure(125000,figsize=(figx, figy))
        plt.subplot(subpl1,subpl2,subpl3)
        plt.figure(135000,figsize=(figx, figy))
        plt.subplot(subpl1,subpl2,subpl3)
        print(i,np.shape(total),color,len(filenames),color_ind)
        plot_itor(total[i],subpl3,color,filenames[color_ind])
        #plot_chronos(total[i],(i%len(freqs))+1,color,filenames[color_ind])
        plot_temperatures(total[i],subpl3,color,filenames[color_ind])
        plot_nFIR(total[i],subpl3,color,filenames[color_ind])
        plot_centroid(total[i],subpl3,color,filenames[color_ind])
        if '2T' in filenames[color_ind] and 'PSI' in filenames[color_ind]:
            #plot_powers(total[i],filenames[color_ind],directory)
            #plot_navg(total[i],subpl3,color,filenames[color_ind])
            #plot_pressure(total[i],subpl3,color,filenames[color_ind])
            #plot_power_balance(total[i],subpl3,filenames[color_ind])
            plot_all_heat_flows(total[i],colors2T[T2_ind],filenames[color_ind],directory)
            T2_ind = T2_ind + 1
            toroidal_modes_sp(total[i],3)
if __name__ == '__main__':
    analysis()
