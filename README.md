# Navigation

The Navigation software is designed to help buyers find a fair price for the scarce commodities in a market following the CRISDIS project guidelines. 
The guidelines are the following. For a given scarce ressource R, each buyers must provide its true need for R. Given the claims and the total supply provided by the sellers, items of rights are fairly distributed among buyers. These items of rights can either be sold to earn money, or used along with money to buy items of good. In the following we are looking for the fair price for R. 
Two different algorithms can be chosen by the user: A machine learning algorithm, more reliable but time consuming to run, and a polynomial algorithm similar to the one described in (Anetta Jedlickova, Martin Loebl, David Sychrovsky, 2021), approximating the equilibrium price by assuming that the price of the right is equal to the price of the good. An exact version of the algorithm is provided in the MIP.py file but it is not used because of its computation time.

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

The input path should be a json file having the following hierarchy: 
An array "productDemand":
    -"amount": number,
    -"credit": number
An array "productSupply":
    -"amount": number,
    -"price": number


For an example, see the test.json file in the data folder.