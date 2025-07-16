print("this is one line \
This is another line \
This is the last line")


parser.add_argument(
    '--is_big_dataset', '-b', 
    type=bool, default=False, 
    help='Is the dataset big? If so, the mAP is calculated every 5 epochs. \
          The rest is 0. I want to exclude the zeros')
