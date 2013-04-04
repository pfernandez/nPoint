#!/bin/bash
set -o errexit

# To background this script, use the command "nohup ./runit.sh > runit.out 2>&1 &".


sourcefile=nPoint.cu
runfile=nPoint


# Set to "true" to display GPU data. Requires a compile copy of deviceQuery from the CUDA SDK
# to be located in the same directory as this script.
devicequery=false  
    
# Set to "true" to save output and data files to a unique directory, named using the current
# timestamp.
savedata=false    

# Set to "true" to save images video to a unique directory. Requires gnuplot for image generation,
# avconv for video conversion, and vlc for video playback. $savedata must also be set to "true".
visualize=false


# Choose a plot style of either "points" or "mesh". $visualize must be set to "true". Using "mesh"
# is slower and requires more data points for accuracy, and either may require adjusting the
# Gnuplot zrange below.
plotstyle="points"


# Compile:
echo "Compiling..."
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
nvcc -DDEBUG -arch=sm_21 $sourcefile -o $runfile  # GeForece GT 540M (defaults to 64-bit.)


[ $devicequery == true ] && ./deviceQuery

if [ $savedata == true ]; then

    # Create directories:
    echo "Creating directories..."
    starttime=$(date +%Y-%m-%d-%H-%M-%S)
    savepath=$(pwd)/$runfile-$starttime
    datapath=$savepath/data
    imagepath=$savepath/images
    mkdir -p $datapath $imagepath

    # Back up code:
    echo "Copying $sourcefile to $savepath/..."
    cp $sourcefile $savepath/

    # Run simulation:
    echo -e "Running simulation...\n"
    ./$runfile $datapath | tee $savepath/info.out


    if [ $visualize == true ]; then

        # Plot data points and save as images:
        echo "Plotting data points..."
        validplot=true
        num=1

        if [ $plotstyle == "mesh" ]; then

            for datfile in $(ls $datapath | grep .dat | sort -g); do
	            lim=$(grep limit $datapath/$datfile | cut -c 11-)
	            plotCommand="
                set terminal jpeg large size 1200,900 crop\n
                set output '${imagepath}/$(printf %05d $num).jpg'\n
                set xrange [-$lim:$lim]\n
                set yrange [-$lim:$lim]\n
                set zrange [-0.2:0.2]\n
                set xyplane at 0\n
                set zeroaxis\n
                set zzeroaxis\n
                unset xtics\n
	            unset ytics\n
	            set ztics axis border offset graph 0,graph 0.17 nomirror tc rgb 'white'\n
                set palette defined (0 'purple',1 'cyan',2 'white')\n
                set colorbox user noborder origin .9,.3 size .01,.4\n
                set autoscale cb\n
                set cblabel 'Psi' tc rgb 'white'\n
                set cbtics tc rgb 'white'\n
	            set object 1 rect from screen 0, 0, 0 to screen 1, 1, 0 behind\n
	            set object 1 rect fc rgb 'black' fillstyle solid 1.0\n
	            unset border\n
	            set view ,,\n
	            set dgrid3d 40,40,1\n
	            set style data lines\n
                set hidden3d\n
                set label at screen .73,.93,0 front 't = ${datfile%%.dat}' tc rgb 'white'\n
                splot '$datapath/$datfile' using 1:2:3 palette notitle\n
                "
	            echo -e $plotCommand | gnuplot
	            num=$(($num+1))
            done            
        
        elif [ $plotstyle == "points" ]; then
        
            for datfile in $(ls $datapath | grep .dat | sort -g); do
	            lim=$(grep limit $datapath/$datfile | cut -c 11-)
                plotCommand="
                set terminal jpeg large size 1200,900 crop\n
                set output '${imagepath}/$(printf %05d $num).jpg'\n
                set xrange [-$lim:$lim]\n
                set yrange [-$lim:$lim]\n
                set zrange [-2:2]\n
                set xyplane at 0\n
                set zeroaxis\n
                set zzeroaxis\n
                unset xtics\n
	            unset ytics\n
	            set ztics axis border offset graph 0,graph 0.17 nomirror tc rgb 'white'\n
                set palette defined (0 'purple',1 'cyan',2 'white')\n
                set colorbox user noborder origin .9,.3 size .01,.4\n
                set autoscale cb\n
                set cblabel 'Psi' tc rgb 'white'\n
                set cbtics tc rgb 'white'\n
	            set object 1 rect from screen 0, 0, 0 to screen 1, 1, 0 behind\n
	            set object 1 rect fc rgb 'black' fillstyle solid 1.0\n
	            unset border\n
	            # set view equal_axes xyz\n
	            set view ,,\n
                set label at screen .73,.93,0 front 't = ${datfile%%.dat}' tc rgb 'white'\n
                splot '$datapath/$datfile' using 1:2:3 with points pointtype 7 pointsize .4 
                palette notitle\n
                "
                echo -e $plotCommand | gnuplot
                num=$(($num+1))
            done 
            
	        echo -e $plotCommand | gnuplot
	        num=$(($num+1))
        
        else
            echo "Invalid plot style."
            validplot=false
        fi

        if [ $validplot == true ]; then
        
            # Create movies from images:
            echo "Creating movie..."
            avconv -i ${imagepath}/%05d.jpg -r 10 $savepath/$plotstyle.avi > /dev/null 2>&1

            # Play movie
            vlc --loop $savepath/$plotstyle.avi > /dev/null 2>&1
        fi
    fi
else

    # Run simulation:
    echo "Running simulation..."
    ./$runfile

fi
