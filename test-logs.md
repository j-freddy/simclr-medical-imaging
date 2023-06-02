## Test accuracy logs

                                     natural / simple / novel
- downstream-pathmnist-100-samples   0.68510 / 0.646  / 0.637
- downstream-lr                      0.70125 / 0.692  / 0.730
- baseline-pathmnist-100-samples     0.39471
- downstream-pathmnist-250-samples   0.76170 / 0.777  / 0.758
- downstream-lr                      0.78454 / 0.787  / 0.813
- baseline-pathmnist-250-samples     0.55864
- downstream-pathmnist-1000-samples  0.82869 / 0.831 / 0.839
- downstream-lr                      0.82967 / 0.834 / 0.842
- baseline-pathmnist-1000-samples    0.65808
- downstream-pathmnist               0.87479 / 0.868 / 0.8721
- downstream-lr                      0.00000 / TODO  / TODO
- baseline-pathmnist-18              0.85738
                                                               path-derma / path
- downstream-dermamnist-100-samples  0.67382 / 0.682 / 0.680 / 0.454      / 0.442
- downstream-lr                      0.67681 / 0.668 / 0.678 / 0.424      / 0.427
- baseline-dermamnist-100-samples    0.66883
- downstream-dermamnist-250-samples  0.70075 / 0.709 / 0.704 / 0.637      / 0.590
- downstream-lr                      0.70025 / 0.698 / 0.702 / 0.495      / 0.496
- baseline-dermamnist-250-samples    0.67930
- downstream-dermamnist-1000-samples 0.74065 / 0.735 / 0.744 / 0.653      / 0.600
- downstream-lr                      0.71920 / 0.714 / 0.741 / 0.575      / 0.546
- baseline-dermamnist-1000-samples   0.71222
- downstream-dermamnist              0.78703 / 0.780 / 0.790
- downstream-lr                      0.73865 / 0.741 / 0.771
- baseline-dermamnist-18             0.75910

- downstream-bloodmnist-100-samples  0.73195 / 0.730 / 0.762
- downstream-lr                      0.77053 / 0.749 / 0.771
- baseline-bloodmnist-100-samples    0.61824
- downstream-bloodmnist-250-samples  0.79567 / 0.793 / 0.818
- downstream-lr                      0.83484 / 0.809 / 0.833
- baseline-bloodmnist-250-samples    0.72786
- downstream-bloodmnist-1000-samples 0.88249 / 0.877 / 0.887
- downstream-lr                      0.88015 / 0.862 / 0.875
- baseline-bloodmnist-1000-samples   0.85238
- downstream-bloodmnist              0.95615 / 0.955 / 0.956
- downstream-lr                      0.91114 / 0.890 / 0.912
- baseline-bloodmnist-18             0.95235

                            retina / path  / path-retina
- downstream-retina-100     0.457  / 0.452 / 0.452
- downstream-lr             0.460  / 0.488 / 0.455
- baseline-retina-100       0.458
- downstream-retina-250     0.455  / 0.455 / 0.450
- downstream-lr             0.463  / 0.530 / 0.498
- baseline-retina-250       0.498
- downstream-retina-1000    0.468  / 0.480 / 0.477
- downstream-lr             0.510  / 0.522 / 0.477
- baseline-retina-1000      0.480

                                     novel greyscale
- downstream-tissue-100-samples      0.459
- downstream-lr                      0.486
- baseline-tissue-100-samples        0.403
- downstream-tissue-250-samples      0.487
- downstream-lr                      0.517
- baseline-tissue-250-samples        0.437
- downstream-tissue-1000-samples     0.513
- downstream-lr                      0.546
- baseline-tissue-1000-samples       0.463
- downstream-tissue                  0.636
- downstream-lr                      
- baseline-tissue-18                 0.637

                                     novel greyscale
- downstream-oct-100-samples         0.570
- downstream-lr                      0.665
- baseline-oct-100-samples           0.380
- downstream-oct-250-samples         0.637
- downstream-lr                      0.658
- baseline-oct-250-samples           0.475
- downstream-oct-1000-samples        0.663
- downstream-lr                      0.708
- baseline-oct-1000-samples          0.601
- downstream-oct                     0.736
- downstream-lr                      
- baseline-oct-18                    0.723

## Test AUROC logs

                                     natural / simple / novel
- downstream-pathmnist-100-samples   0.909   / 0.883  / 0.884
- downstream-lr                      0.939   / 0.943  / 0.957
- baseline-pathmnist-100-samples     0.780
- downstream-pathmnist-250-samples   0.921   / 0.939  / 0.931
- downstream-lr                      0.965   / 0.965  / 0.975
- baseline-pathmnist-250-samples     0.839
- downstream-pathmnist-1000-samples  0.950   / 0.955  / 0.959
- downstream-lr                      0.977   / 0.979  / 0.982
- baseline-pathmnist-1000-samples    0.879
- downstream-pathmnist               0.964   / 0.973  / 0.972
- downstream-lr                      
- baseline-pathmnist-18              0.974

                                                               path-derma / path
- downstream-dermamnist-100-samples  0.756   / 0.741 / 0.741 / 0.741      / 0.707
- downstream-lr                      0.735   / 0.759 / 0.757 / 0.795      / 0.823
- baseline-dermamnist-100-samples    0.725
- downstream-dermamnist-250-samples  0.839   / 0.846 / 0.843 / 0.706      / 0.750
- downstream-lr                      0.811   / 0.840 / 0.846 / 0.866      / 0.767
- baseline-dermamnist-250-samples    0.792
- downstream-dermamnist-1000-samples 0.871   / 0.871 / 0.879 / 0.776      / 0.776
- downstream-lr                      0.873   / 0.882 / 0.896 / 0.803      / 0.850
- baseline-dermamnist-1000-samples   0.829
- downstream-dermamnist              0.925   / 0.926 / 0.926
- downstream-lr                      0.907   / 0.909 / 0.930
- baseline-dermamnist-18             0.904

- downstream-bloodmnist-100-samples  0.934   / 0.931 / 0.946
- downstream-lr                      0.957   / 0.943 / 0.958
- baseline-bloodmnist-100-samples    0.890
- downstream-bloodmnist-250-samples  0.953   / 0.949 / 0.962
- downstream-lr                      0.970   / 0.961 / 0.971
- baseline-bloodmnist-250-samples    0.919
- downstream-bloodmnist-1000-samples 0.982   / 0.981 / 0.983
- downstream-lr                      0.984   / 0.981 / 0.985
- baseline-bloodmnist-1000-samples   0.973
- downstream-bloodmnist              0.997   / 0.996 / 0.996
- downstream-lr                      0.992   / 0.989 / 0.992
- baseline-bloodmnist-18             0.996

                            retina / path  / path-retina
- downstream-retina-100     0.664  / 0.644 / 0.632
- downstream-lr             0.660  / 0.677 / 0.683
- baseline-retina-100       0.648
- downstream-retina-250     0.663  / 0.651 / 0.644
- downstream-lr             0.658  / 0.686 / 0.686
- baseline-retina-250       0.630
- downstream-retina-1000    0.667  / 0.680 / 0.695
- downstream-lr             0.697  / 0.726 / 0.688
- baseline-retina-1000      0.668

                                     novel greyscale
- downstream-tissue-100-samples      0.725
- downstream-lr                      0.778
- baseline-tissue-100-samples        0.648
- downstream-tissue-250-samples      0.761
- downstream-lr                      0.812
- baseline-tissue-250-samples        0.682
- downstream-tissue-1000-samples     0.790
- downstream-lr                      0.853
- baseline-tissue-1000-samples       0.728
- downstream-tissue                  0.889
- downstream-lr                      
- baseline-tissue-18                 0.917

                                     novel greyscale
- downstream-oct-100-samples         0.786
- downstream-lr                      0.899
- baseline-oct-100-samples           0.659
- downstream-oct-250-samples         0.851
- downstream-lr                      0.915
- baseline-oct-250-samples           0.755
- downstream-oct-1000-samples        0.857
- downstream-lr                      0.909
- baseline-oct-1000-samples          0.818
- downstream-oct                     0.903
- downstream-lr                      
- baseline-oct-18                    0.901
