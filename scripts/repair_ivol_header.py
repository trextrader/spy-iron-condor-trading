import os

file_path = "data/ivolatility/spy_options_ivol_1year.csv"
header = "symbol,exchange,date,underlying_price,option_symbol,expiration,strike,call_put,style,ask,bid,mean_price,volume,open_interest,iv,iv_raw,delta,gamma,vega,theta,rho,oi_2,option_id,trade_date_fetch\n"

if not os.path.exists(file_path):
    print("File not found")
    exit()

with open(file_path, 'r') as f:
    lines = f.readlines()

if not lines:
    print("File empty")
    exit()

# Check if header exists
if lines[0].startswith("symbol"):
    print("Header already exists")
else:
    print(f"Prepending header to {len(lines)} rows")
    lines.insert(0, header)
    with open(file_path, 'w') as f:
        f.writelines(lines)
    print("Done")
