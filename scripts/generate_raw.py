import re, sys, os

def process_dir (indir) :
    """Process the input directory files.
    """
    for fname in os.listdir (indir) :
        fpath = os.path.join (indir, fname)
        with open (fpath) as ifp :
            from_sent = []
            to_sent = []
            for idx,line in enumerate (ifp) :
                parts = re.split (r"\t", line.strip ())
                if len (parts) == 2 and '<eos>' in line :
                    print (" ".join (from_sent))
                    print (" ".join (to_sent).lower (),file=sys.stderr)
                    from_sent = []
                    to_sent = []
                else :
                    parts [2] = re.sub (r"_letter", "", parts [2])
                    from_sent.append (parts [1])
                    if parts [2] == 'sil' :
                        continue
                    
                    if parts [2] == '<self>' :
                        to_sent.append (parts [1])
                    else :
                        to_sent.append (parts [2])
    return

if __name__ == "__main__" :
    import argparse

    process_dir (sys.argv [1])
