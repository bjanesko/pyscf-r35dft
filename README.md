PySCF implementation of rung-3.5 density functionals and interpretive tools. 

Density functionals are implemented as r35dft, an extension of the dft module 

Density functionals include the Minnesota functional M11po. The semilocal piece
of this functional is implemented in a fork of libxc available on the gitlab
repository bjanesko/libxc-r35dft . Use XC functionals xc='HYB_MGGA_X_M11PO,MGGA_C_M11PO' 

Int
=========================

2026-05-08

* Version 0.1

Install
-------
* Install to python site-packages folder
```
pip install git+https://github.com/bjanesko/pyscf-r35dft 
```

* Install in a custom folder for development
```
git clone https://github.com/bjanesko/pyscf-r35dft /home/abc/local/path

# Set pyscf extended module path
echo 'export PYSCF_EXT_PATH=/home/abc/local/path:$PYSCF_EXT_PATH' >> ~/.bashrc
```

You can find more details of extended modules in the document
[extension modules](http://pyscf.org/pyscf/install.html#extension-modules)
