# Navigation

The Navigation software is designed to help buyers find a fair price for the scarce commodities in a market following the CRISDIS project guidelines. 
The guidelines are the following. For a given scarce good G, each buyers must provide its true need for G. Given the claims and the total supply provided by the sellers, items of rights are fairly distributed among buyers. These items of rights can either be sold to earn money, or used along with money to buy items of good. In the following we are looking for the fair price for G. 
Two different algorithms can be chosen by the user: A machine learning algorithm, more reliable but time consuming to run, and a polynomial algorithm similar to the one described in (Aneta Jedlickova, Martin Loebl, David Sychrovsky, 2021 [arxiv](https://arxiv.org/pdf/2207.00898.pdf)), approximating the equilibrium price by assuming that the price of the right is equal to the price of the good. An exact version of the algorithm is provided in the MIP.py file but it is not used because of its computation time.

Required:
    Python 3.8

Python packages:
    datetime >= 4.8,
    MIP >= 1.14,
    Numpy >= 1.23,
    tensorflow >= 2.4,
    matplotlib >= 3.2,
    tensorflow_probability >= 0.12.


usage: python main.py --path __path__ --mode __mode__

- __path__ is the path to the input json file.
- __mode__ is either "advice", which will run the equilibrium algorithm and return the adviced price for the commodity,
or "predict", which will run the machine learning program to predict the prices according to the json file.

The input path should be a json file having the following hierarchy: 
An array "productDemand":
    -"amount": number,
    -"credit": number
An array "productSupply":
    -"amount": number,
    -"price": number


For an example, see the test.json file in the data folder.
