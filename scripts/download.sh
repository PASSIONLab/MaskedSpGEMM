#!/bin/bash

M=("amazon0505" "2cubes_sphere" "rma10" "pwtk" "majorbasis" "scircuit" "patents_main" "m133-b3" "mc2depi" "web-Google" "offshore" "cage12")
URLs=(
    "https://sparse.tamu.edu/MM/SNAP/amazon0505.tar.gz"
    "https://sparse.tamu.edu/MM/Um/2cubes_sphere.tar.gz"
    "https://sparse.tamu.edu/MM/Bova/rma10.tar.gz"
    "https://sparse.tamu.edu/MM/Boeing/pwtk.tar.gz"
    "https://sparse.tamu.edu/MM/QLi/majorbasis.tar.gz"
    "https://sparse.tamu.edu/MM/Hamm/scircuit.tar.gz"
    "https://sparse.tamu.edu/MM/Pajek/patents_main.tar.gz"
    "https://sparse.tamu.edu/MM/JGD_Homology/m133-b3.tar.gz"
    "https://sparse.tamu.edu/MM/Williams/mc2depi.tar.gz"
    "https://sparse.tamu.edu/MM/SNAP/web-Google.tar.gz"
    "https://sparse.tamu.edu/MM/Um/offshore.tar.gz"
    "https://sparse.tamu.edu/MM/vanHeukelum/cage12.tar.gz"
)

mkdir -p ./assets
counter=0
for url in ${URLs[@]}; do
    wget $url
    tar -xzf ${M[$counter]}".tar.gz"
    mv ./${M[$counter]}/${M[$counter]}".mtx" ./assets/${M[$counter]}".mtx"
    rm -rf ${M[$counter]}*
    ((counter++))
done

