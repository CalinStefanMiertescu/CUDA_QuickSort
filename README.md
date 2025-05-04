# Quicksort cu CUDA – Sortare rapidă pe GPU

Acest proiect este o implementare a algoritmului Quicksort folosind CUDA, gândit special pentru sortarea eficientă a unor seturi mari de numere. În loc să folosim doar CPU-ul, profităm de puterea paralelă a plăcii video (GPU) pentru a accelera procesul de sortare.

## De ce CUDA?
Quicksort e cunoscut pentru performanțele lui pe CPU, dar devine o provocare interesantă când vrei să îl adaptezi pentru rulare paralelă. Cu CUDA putem paraleliza partiționarea și alte operații, câștigând timp serios atunci când sortăm milioane de elemente.

## Tehnologii utilizate
- C / C++
  
- CUDA Toolkit
  
- NVIDIA GPU (compatibil cu CUDA)

## Conținutul proiectul
- Codul sursă pentru Quicksort pe CPU și CUDA

- Funcții pentru citirea datelor din fișier

- Compararea performanței între versiunea clasică și cea paralelă

- Scripturi de build (CMake)
