# BigMart sales prediction

## Description of the task:
"The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and find out the sales of each product at a particular store.
Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales."

You can find the data set [here](www.kaggle.com%2Fbrijbhushannanda1979%2Fbigmart-sales-data)

You can check my post on Medium [here](https://medium.com/p/2301935c0948)

## Attributes
1. Item_Identifier: code to identify the product.
2. Item_Weight: weight of the product.
3. Item_Fat_Content: classification of the product based on its fat content.
4. Item_Visibility: measure that refers to the knowledge of the product
5. product on the consumer. How easy can the product be found?
6. Item_MRP: maximum retailed price. Price calculated by the manufacturer indicating the highest price that can be charged for the product.
7. Outlet_Identifier: store identifier.
8. Outlet_Establishment_Year: store oppening year.
9. Outlet_Size: store size.
10. Outlet_Location_Type: classification of stores according to location.
11. Outlet_Type: type of store.
12. Item_Outlet_Sales: product sales in each observation.

## Files
On BigMart-sales.ipynb you can find the general code where I explored the attributers and train a LightGBM classifier.
On Preprocessing-function.py you can find the function that I used to realize all the preprocessing that I explored on the previous file.
