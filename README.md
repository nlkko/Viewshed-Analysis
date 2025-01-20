# Viewshed Analysis with Ridge Detection and Visualization on Digital Elevation Models

Master's thesis in Computer Science at NTNU 
<br><br>
Written by:
<ul>
  <li>Nikola Dordevic</li>
  <li>Jørund Topp Løvlien</li>
</ul>

Supervisor:
<ul>
  <li>Theoharis Theoharis</li>
</ul>

Link to [thesis](https://drive.google.com/file/d/1hxsbDvyRq4aARuwwidbSg6qlV1obYejl/view)

## Abstract

In this thesis we examine the different viewshed analysis algorithms, namely
R2, R3 and Reference planes. Using a ridge detection algorithm known as
the Steepest ascent method, we extract ridge data from digital elevation
models. The ridge data is then used together with the R2 and R3 algorithms
to test whether this can provide speed ups. Along with this we implemented
an rudimentary app to visualize the resulting viewsheds. In the end we
found that using ridge data indeed does give speed up with the R2 algorithm.
When it comes to the R3 algorithm with ridge data we found that our
implementation proves to be much slower, which should be investigated
further.

## Results
<br><br>
<p align="center" width="100%">
    <img width="70%" src="/report/visualization.png"> </br>
    Visualization of a viewshed analysis using the R3 Algorithm.
</p>

<br><br>

The resulting viewshed at 63.4162 degrees latitude, 10.3982 degrees longitude.

<table style="width: 100%; border-collapse: separate; border-spacing: 20px;">
    <tr>
        <td style="width: 30%; text-align: center;">
            <figure>
                <img src="/report/r2.png"
                    style="width: 100%; height: auto;"
                    alt="Brezovica">
                <figcaption>R2 Algorithm</figcaption>
            </figure>
        </td>
        <td style="width: 30%; text-align: center;">
            <figure>
                <img src="/report/r3.png"
                    style="width: 100%; height: auto;"
                    alt="Brezovica">
                <figcaption>R3 Algorithm</figcaption>
            </figure>
        </td>
        <td style="width: 30%; text-align: center;">
            <figure>
                <img src="/report/ref_planes.png"
                    style="width: 100%; height: auto;"
                    alt="Brezovica">
                <figcaption>Reference Planes Algorithm</figcaption>
            </figure>
        </td>
    </tr>
</table>


# Installation
Install dependencies: <br />
```conda install -f environment.yml```

Optional dependency installation for Anaconda environment: <br />
```conda install -c conda-forge geopandas``` <br />
```conda install -c conda-forge rasterio --freeze-installed``` <br />

Controls for inspection tool in Panda3D: <br />
https://docs.panda3d.org/1.10/python/debugging/inspection-tools/enhanced-mouse-navigation
