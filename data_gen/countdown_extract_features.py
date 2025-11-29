import pandas as pd
import ast
import math
from itertools import combinations

df = pd.read_csv('/mnt/data/countdown_converted.csv')

def is_prime(n):
    if n<2: return False
    if n%2==0: return n==2
    r=int(n**0.5)
    for i in range(3,r+1,2):
        if n%i==0: return False
    return True

def max_depth(expr):
    if not isinstance(expr,str): return 0
    d=0;m=0
    for c in expr:
        if c=="(":
            d+=1;m=max(m,d)
        elif c==")":
            d-=1
    return m

def op_counts(expr):
    if not isinstance(expr,str): return (0,0,0,0)
    return expr.count("+"),expr.count("-"),expr.count("*"),expr.count("/")

def easy_pairs(nums):
    e=0
    for a,b in combinations(nums,2):
        if (a+b)%10==0: e+=1
        if min(a,b)!=0 and max(a,b)%min(a,b)==0: e+=1
    return e

rows=[]
for _,r in df.iterrows():
    nums = ast.literal_eval(r['numbers']) if isinstance(r['numbers'],str) else []
    tgt = int(r['target']) if not pd.isna(r['target']) else 0
    sol = r['solution'] if isinstance(r['solution'],str) else ""

    add,sub,mul,div = op_counts(sol)

    rows.append({
        "id": r["id"],
        "n_numbers": len(nums),
        "range": max(nums)-min(nums) if nums else 0,
        "std": pd.Series(nums).std() if nums else 0,
        "count_small": sum(x<10 for x in nums),
        "count_large": sum(x>50 for x in nums),
        "count_duplicates": len(nums)-len(set(nums)),
        "count_even": sum(x%2==0 for x in nums),
        "count_odd": sum(x%2!=0 for x in nums),
        "count_div_2": sum(x%2==0 for x in nums),
        "count_div_3": sum(x%3==0 for x in nums),
        "count_div_5": sum(x%5==0 for x in nums),
        "count_div_7": sum(x%7==0 for x in nums),
        "count_primes": sum(is_prime(x) for x in nums),
        "distance_simple": abs(sum(nums)-tgt) if nums else 0,
        "distance_max": abs(max(nums)-tgt) if nums else 0,
        "distance_avg": abs(sum(nums)/len(nums)-tgt) if nums else 0,
        "easy_pairs": easy_pairs(nums),
        "log_target": math.log10(abs(tgt)) if tgt!=0 else 0,
        "expr_depth": max_depth(sol),
        "count_add": add,
        "count_sub": sub,
        "count_mul": mul,
        "count_div": div,
        "noncomm_ops": sub+div,
        "numbers": r["numbers"],
        "target": r["target"],
        "solution": r["solution"]
    })

out = pd.DataFrame(rows)
path = '/mnt/data/countdown_features.csv'
out.to_csv(path, index=False)
