# SympNetsProcessing

Iekļauti visi nepieciešamie kodi, lai pēc maģistra darba "Simplektiskas skaitliskās metodes ar struktūru saglabājošu neirona tīkla apstrādi" aprakstītās metodes, varētu trenēt un veikt skaitlisku kļūdu analīzi apstrādes metodēm, kas veiktas balstoties uz simplektisku gradientu neironu tīklu apstrādi.

# Pirms skriptu failu lietošanas
## Lejupielāde
Šos failus iespējams lejupielādēt kā .zip failu, uzspiežot uz zaļās koda pogas. Alternatīvi, ja ir ieinstalēta _GitHub_ pakotne, tad to iespējams izradīt lietojot termināli ar komandu _git clone https://github.com/DavisKalvans/SympNetsProcessing_.

## Python vides uzstādīšana
Darbs tika izstrādāts ar _Python_ versiju _3.8.8_. Iespējams arī jaunākas versijas strādās, ja pārējo pakotņu konkrētās versijas būs saderīgas. Ar failu _requirements.txt_ nepieciešams izveidot _Python virtuālo vidi_, piemēram, ar _Conda_ un _Spyder_, _Visual Studio Code_ vai caur termināli (ja ir ieinstalēta _virtualenv_ pakotne).

## Mapju struktūras izmantošana
Nepieciešams palaist scripta failus _/OneTrajectory/folderStructure.py_ un _/RandomPoints/folderStructure.py_, lai tiktu izveidota mapju struktūra uztrenēto neirona tīklu un ģenerēto grafiku saglabāšanai.

# Īsa pamācība
Mapē _OneTrajectory_ atrodami visi skripti modeļiem, kuri tiek trenēti un validēti ar randomizētiem Hamiltona sistēmas punktiem kādā apgabalā. Mapē _RandomPoints_ atrodami visi skripti modeļiem, kuri tiek trenēti un validēti ar vienu Hamiltona sistēmas trajektoriju. Sekojošās instrukcijas ir identiskas abos gadījumos.

# Trenēšanas un validācijas datu kopu ģenerēšana
Visas darbā izmantotās datu kopas jau ir iekļautas. Lai ģenerētu jaunas ar citu trenēšanas punktu skaitu $M$, validācijas punktu skaitu $N$ vai laika soli $h$, tad nepieciešams izmantot attiecīgās problēmas _ _dataRand.py_ scriptus, kuri atrodas _TrainingData_ mapē.

# Modeļu trenēšana un to prognožu kļūdu un konverģences grafiku iegūšana
Nesimetrisku modeļu gadījumā ($k=1$) imantot _training.py_ un simetrisku modeļu gadījumā ($k=2$) izmantot _trainingSym.py_. Iespējams nomainīt visas parametru vērtības (slāņu skaita, biezuma, izmantoto trenēšanas un validācijas datu kopu, prognožu ilgumu utt.), kā arī trenēt vairākus modeļus pēc kārtas, lietojot sarakstus (lists) ar vērtībām, un iegūt 
* modeli; 
* tā trenēšanas kļūdu grafiku; 
* progonožu kļūdu grafikus konkrētai trajektorijai ar specifizētu sākumpunktu $x_0$; 
* prognožu kļūdu konverģences grafikus konkŗetai trajektorijai ar specifizētu sākumpunktu $x_0$;
* progonožu vidējo kļūdu grafikus vairākām trajektorijām skaitā _nr_trajects_;
* progonožu vidējās konverģences grafikus vairākām trajektorijām skaitā _nr_trajects_.

# Trenētu modeļu prognožu vidējās kļūdas, konverģence un VPT
Lai analizētu vairākus jau uztrenētus modeļus attiecībā pret slāņu skaita $L$, biezuma $n$ un dažādām svaru inicializācijām $nM$, lieto skriptu _analysis.py_. Tas automātiski veiks nepieciešamos aprēķinu un prognozes, rezultātā dodot tabulas, kuras arī tiek saglabātas $LaTeX$ formātā
* modeļu trenēšanas un validācijas kļūdām un to standartnovirzēm;
* kodola un apstrādes metožu (kas iegūtas ar šiem modeļiem) prognožu kļūdas un standartnovirzes līdz laikam _Tend_ un trajektorijām skaitā _nr_trajects_;
* kodola un konnkrētas apstrādes metodes (kas iegūta ar šiem modeļiem) VPT (valid prediction time) līdz laikam _Tend_ un trajektorijām skaitā _nr_trajects_.
