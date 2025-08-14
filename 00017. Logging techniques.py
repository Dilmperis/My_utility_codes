# 1. Get the exact time of the loggings:

import time

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log(f'Example of the log function (gives you time in Hour:Minute:Second format)')
log(f'You can use symbols like:     └─ scanned 20 files so far')

'''
Example output: 
      [15:42:18] Example of the log function (gives you time in Hour:Minute:Second format)
      [15:42:18] You can use symbols like:     └─ scanned 20 files so far  
'''


# 2. Symbols you can use:
'''
└─ , 
'''
