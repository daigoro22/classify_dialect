import argparse
import json
import itertools

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--pref_file',type=str,default='corpus/all_pft.txt')
    parser.add_argument('-at','--area_table_file',type=str,default='corpus/area.json')
    parser.add_argument('-a','--area_file',type=str,default='corpus/all_area.txt')
    args = parser.parse_args()

    with open(args.area_table_file,'r') as f:
        area_to_pref_dict = json.load(f)

    with open(args.pref_file,'r') as f:
        corpus = [c.strip() for c in f]
    
    pref_list = list(set(corpus))
    pref_list_area = list(itertools.chain.from_iterable(area_to_pref_dict.values()))

    #assertion
    for p in pref_list_area:
        try:
            assert(p in pref_list)
        except AssertionError:
            raise AssertionError(p)

    pref_to_area_dict_list = [{p:area for p in pref} for area,pref in area_to_pref_dict.items()]
    pref_to_area_dict = {}
    for pta_dict in pref_to_area_dict_list:
        pref_to_area_dict.update(pta_dict)

    with open(args.area_file,'w') as f:
        for c in corpus:
            if c in pref_to_area_dict:
                f.write(pref_to_area_dict[c]+'\n')
            else:
                f.write(c+'\n')