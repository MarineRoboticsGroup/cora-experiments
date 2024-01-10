from matlab_interfaces import export_fg_to_matlab_cora_format

if __name__ == "__main__":
    fpath = "/home/alan/test.pyfg"
    from py_factor_graph.io.pyfg_text import read_from_pyfg_text

    fg = read_from_pyfg_text(fpath)

    new_fpath = "/home/alan/test.mat"
    export_fg_to_matlab_cora_format(fg, new_fpath)
