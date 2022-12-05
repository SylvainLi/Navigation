# Navigation

The Navigation software is designed to help buyers find a fair price for the scarce commodities in a market following the CRISDIS project guidelines. Two different algorithms can be chosen by the user: A machine learning algorithm, more reliable but time consuming to run, and an algorithm similar to the one described (Anetta Jedlickova, Martin Loebl, David Sychrovsky, 2021) approximating the equilibrium price by assuming the price of the right is equal the price of the good. An exact version of the algorithm is provided in the MIP.py file but it is not used because of its speed.

Required:
    Python 3.8

Python packages:
    datetime
    MIP
    Numpy
    tensorflow
    matplotlib
    pickle
    tensorflow_probability


usage: python name.py --path <path> --mode <mode>

<path> is the path to the input json file.
<mode> is either "advice", which will run the equilibrium algorithm and return the adviced price for the commodity,
or "predict", which will run the machine learning program to predict the prices according to the json file.
