#!/bin/sh

cd data/Raw
mkdir ./images

# download irradiance measurements
echo "Downloading Folsom_irradiance.csv ..." && \
    wget https://zenodo.org/records/2826939/files/Folsom_irradiance.csv && \
    echo "Folsom_irradiance.csv downloaded"

# download sky images
echo "Downloading and extracting Folsom_sky_images_2014.tar.bz2 ..." && \
    # wget https://zenodo.org/records/2826939/files/Folsom_sky_images_2014.tar.bz2 && \
    echo "Folsom_sky_images_2014.tar.bz2 downloaded" && \
    tar xf Folsom_sky_images_2014.tar.bz2 --directory ./images/ && \
    echo "Folsom_sky_images_2014.tar.bz2 extracted from archive" && \
    rm Folsom_sky_images_2014.tar.bz2

echo "Downloading and extracting Folsom_sky_images_2015.tar.bz2 ..." && \
    # wget https://zenodo.org/records/2826939/files/Folsom_sky_images_2015.tar.bz2 && \
    echo "Folsom_sky_images_2015.tar.bz2 downloaded" && \
    tar xf Folsom_sky_images_2015.tar.bz2 --directory ./images/ && \
    echo "Folsom_sky_images_2015.tar.bz2 extracted from archive" && \
    rm Folsom_sky_images_2015.tar.bz2

echo "Downloading and extracting Folsom_sky_images_2016.tar.bz2 ..." && \
    # wget https://zenodo.org/records/2826939/files/Folsom_sky_images_2016.tar.bz2 && \
    echo "Folsom_sky_images_2016.tar.bz2 downloaded" && \
    tar xf Folsom_sky_images_2016.tar.bz2 --directory ./images/ && \
    echo "Folsom_sky_images_2016.tar.bz2 extracted from archive" && \
    rm Folsom_sky_images_2016.tar.bz2
