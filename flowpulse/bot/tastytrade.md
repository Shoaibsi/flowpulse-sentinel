tastytrade.yaml 
Instruments
﻿

GET
Cryptocurrencies
https://api.cert.tastyworks.com/instruments/cryptocurrencies
﻿

GET
Cryptocurrency by Symbol
https://api.cert.tastyworks.com/instruments/cryptocurrencies/:symbol
﻿

Path Variables
symbol
BTC/USD
GET
Active Equities
https://api.cert.tastyworks.com/instruments/equities/active
﻿

GET
Equities
https://api.cert.tastyworks.com/instruments/equities?symbol=AAPL
﻿

Query Params
symbol
AAPL
GET
Equity by Symbol
https://api.cert.tastyworks.com/instruments/equities/:symbol
﻿

Path Variables
symbol
AAPL
GET
Equity Options
https://api.cert.tastyworks.com/instruments/equity-options
﻿

GET
Equity Option by Symbol
https://api.cert.tastyworks.com/instruments/equity-options/:symbol
﻿

Path Variables
symbol
GET
Futures
https://api.cert.tastyworks.com/instruments/futures
﻿

GET
Future by Symbol
https://api.cert.tastyworks.com/instruments/futures/:symbol
﻿

Path Variables
symbol
GET
Future Option Products
https://api.cert.tastyworks.com/instruments/future-option-products
﻿

GET
Future Option Product
https://api.cert.tastyworks.com/instruments/future-option-products/:exchange/:root_symbol
﻿

Path Variables
exchange
CME
root_symbol
ES
GET
Future Products
https://api.cert.tastyworks.com/instruments/future-products
﻿

GET
Future Product
https://api.cert.tastyworks.com/instruments/future-products/:exchange/:code
﻿

Path Variables
exchange
CME
code
ES
GET
Quantity Decimal Precisions
https://api.cert.tastyworks.com/instruments/quantity-decimal-precisions
﻿

GET
Warrants
https://api.cert.tastyworks.com/instruments/warrants
﻿

GET
Warrant by Symbol
https://api.cert.tastyworks.com/instruments/warrants/:symbol
﻿

Path Variables
symbol
GET
Future Option Chains by Symbol
https://api.cert.tastyworks.com/futures-option-chains/:symbol
﻿

Path Variables
symbol
ES
GET
Future Option Chains Nested by Symbol
https://api.cert.tastyworks.com/futures-option-chains/:symbol/nested
﻿

Path Variables
symbol
CL
GET
Option Chains by Symbol
https://api.cert.tastyworks.com/option-chains/:symbol
﻿

Path Variables
symbol
AAPL
GET
Option Chains Nested by Symbol
https://api.cert.tastyworks.com/option-chains/:symbol/nested
﻿

Path Variables
symbol
AAPL
GET
Option Chains Compact by Symbol
https://api.cert.tastyworks.com/option-chains/:symbol/compact
﻿

Path Variables
symbol
AAPL
Orders
﻿

POST
Order Dry Run
https://api.cert.tastyworks.com/accounts/:account_number/orders/dry-run
﻿

Path Variables
account_number
Body
raw (json)
View More
json
{
  "order-type": "Limit",
  "price": 1.0,
  "price-effect": "Debit",
  "time-in-force": "Day",
  "legs": [
    {
      "instrument-type": "Equity",
      "action": "Buy to Open",
      "quantity": 100,
      "symbol": "AAPL"
    }
  ]
}
POST
Equity Order
https://api.cert.tastyworks.com/accounts/:account_number/orders
﻿

Path Variables
account_number
Body
raw (json)
View More
json
{
  "order-type": "Limit",
  "price": 1.0,
  "price-effect": "Debit",
  "time-in-force": "Day",
  "legs": [
    {
      "instrument-type": "Equity",
      "action": "Buy to Open",
      "quantity": 100,
      "symbol": "AAPL"
    }
  ]
}
GET
Live Orders
https://api.cert.tastyworks.com/accounts/:account_number/orders/live
Returns all orders relevant to today. This includes any orders that were cancelled today.

Path Variables
account_number
GET
All Orders
https://api.cert.tastyworks.com/accounts/:account_number/orders
﻿

Path Variables
account_number
GET
Order by Id
https://api.cert.tastyworks.com/accounts/:account_number/orders/:order_id
﻿

Path Variables
account_number
order_id
DELETE
Cancel Order
https://api.cert.tastyworks.com/accounts/:account_number/orders/:order_id
﻿

Path Variables
account_number
order_id
PUT
Replace Order
https://api.cert.tastyworks.com/accounts/:account_number/orders/:order_id
﻿

Path Variables
account_number
order_id
Body
raw (json)
View More
json
{
  "order-type": "Limit",
  "price": 1.0,
  "price-effect": "Debit",
  "time-in-force": "Day",
  "legs": [
    {
      "instrument-type": "Equity",
      "action": "Buy to Open",
      "quantity": 100,
      "symbol": "AAPL"
    }
  ]
}
PATCH
Edit Order
https://api.cert.tastyworks.com/accounts/:account_number/orders/:order_id
﻿

Path Variables
account_number
order_id
Body
raw (json)
json
{
  "order-type": "Limit",
  "price": 2.0,
  "price-effect": "Debit",
  "time-in-force": "Day"
}
POST
Edit Order Dry Run
https://api.cert.tastyworks.com/accounts/:account_number/orders/:order_id/dry-run
﻿

Path Variables
account_number
order_id
Body
raw (json)
json
{
  "order-type": "Limit",
  "price": 2.0,
  "price-effect": "Debit",
  "time-in-force": "Day"
}
Symbol Search
﻿

GET
Search for symbol
https://api.cert.tastyworks.com/symbols/search/:symbol
﻿

Path Variables
symbol
AA
Transactions
﻿

GET
Account Transactions
https://api.cert.tastyworks.com/accounts/:account_number/transactions
﻿

Path Variables
account_number
GET
Account Transaction by Id
https://api.cert.tastyworks.com/accounts/:account_number/transactions/:id
﻿

Path Variables
account_number
id
GET
Total Transaction Fees
https://api.cert.tastyworks.com/accounts/:account_number/transactions/total-fees
﻿

Path Variables
account_number
Net Liq History
﻿

GET
Net Liquidating Value History
https://api.cert.tastyworks.com/accounts/:account_number/net-liq/history
﻿

Path Variables
account_number
Market Metrics
﻿

GET
Volatility Data
https://api.cert.tastyworks.com/market-metrics?symbols=AAPL,FB
﻿

Query Params
symbols
AAPL,FB
GET
Dividend History
https://api.cert.tastyworks.com/market-metrics/historic-corporate-events/dividends/:symbol
﻿

Path Variables
symbol
T
GET
Earnings Report History
https://api.cert.tastyworks.com/market-metrics/historic-corporate-events/earnings-reports/:symbol
﻿

Path Variables
symbol
T