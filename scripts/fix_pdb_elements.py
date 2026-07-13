#!/usr/bin/env python3
"""Fill missing element column in PDB ATOM/HETATM records.

Usage: python scripts/fix_pdb_elements.py input.pdb -o fixed.pdb
"""
import argparse
import re


TWO_LETTER_ELEMENTS = {
    'CL', 'BR', 'NA', 'MG', 'AL', 'SI', 'FE', 'ZN', 'CA', 'MN', 'CO', 'CU',
    'NI', 'CD', 'HG', 'SE', 'LI', 'BE', 'NE', 'AR', 'KR', 'XE',
}


def infer_element(atom_name: str, record_name: str = 'ATOM') -> str:
    # Standard protein atom names such as " CA " and " CD " are carbons.
    if record_name == 'ATOM' and atom_name[:1] == ' ':
        for ch in atom_name:
            if ch.isalpha():
                return ch.upper()

    name = atom_name.strip()
    # remove leading digits
    name = re.sub(r"^[0-9]+", "", name)
    if not name:
        return 'C'
    letters = ''.join(ch for ch in name if ch.isalpha()).upper()
    if len(letters) >= 2 and letters[:2] in TWO_LETTER_ELEMENTS:
        return letters[:2].title()
    if letters:
        return letters[0]
    # fallback
    return name[0].upper()


def fix_pdb(in_path: str, out_path: str):
    with open(in_path, 'r') as rf, open(out_path, 'w') as wf:
        for line in rf:
            if line.startswith(('ATOM  ', 'HETATM')):
                if len(line) < 78:
                    line = line.rstrip('\n').ljust(78) + '\n'
                # PDB columns: atom name cols 13-16 (1-based), element cols 77-78
                atom_name = line[12:16]
                element = line[76:78]
                if element.strip() == '':
                    el = infer_element(atom_name, line[:6].strip())
                    # place element right-justified in cols 77-78
                    new_line = line[:76] + el.rjust(2) + line[78:]
                    wf.write(new_line)
                else:
                    wf.write(line)
            else:
                wf.write(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', help='input PDB file')
    p.add_argument('-o', '--out', help='output PDB file', required=True)
    args = p.parse_args()
    fix_pdb(args.input, args.out)


if __name__ == '__main__':
    main()
