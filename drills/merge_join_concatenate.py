from pprint import pprint
import pandas as pd
import myutils as my

# customisations
pd.set_option('display.width', 2000)
pd.set_option('display.max.columns', 100)
x = my.random_df(10, 7, str_frac=0.2)
y = my.random_df(4, 10, row_min=7, col_min=2, str_frac=0.1)

xx = x.copy()
yy = y.copy()

# pandas merge

print("-----Merge-----")
pprint(xx)
pprint(yy)

x_merge_y = pd.merge(xx.rename(columns={4: '4L'}), yy.rename(columns={4: '4R'}),
                     how='right',
                     suffixes=['l', 'r'], left_on=['4L'], right_on=['4R'])

pprint(x_merge_y)

# pandas concat
print("-----Concat-----")
x_concat_y = pd.concat([xx, yy], axis=1, join='outer')
pprint(xx)
pprint(yy)
pprint(x_concat_y)
