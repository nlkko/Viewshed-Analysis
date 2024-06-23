Installing packages for Anaconda environment: <br />
```conda install -c conda-forge geopandas``` <br />
```conda install -c conda-forge rasterio --freeze-installed``` <br />


Export dependencies: <br />
```conda env export --no-builds | findstr -v "prefix" > environment.yml``` <br/>
```conda env export --from-history | findstr -v "prefix" > environment.yml``` <br/>
Only specified libraries

Install dependencies: <br />
```conda install -f environment.yml```

conda install conda-forge::scikit-spatial

Controls for inspection tool in Panda3D: <br />
https://docs.panda3d.org/1.10/python/debugging/inspection-tools/enhanced-mouse-navigation