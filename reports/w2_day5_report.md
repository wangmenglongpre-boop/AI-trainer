# W2 Day 5: temperature sampling + Bigram vs MLP

- split seed=42, names train/dev/test = 25626/3203/3204; same vocab and targets (incl. end boundary)
- MLP: context=3, emb=10, hidden=128, Adam lr=0.001, batch=32, best step=16000
- sampling: prompts=['', 'a', 'ma', 'jo', 'kay'], 10 names per prompt, max 20 new chars, seed=42

| model | train | dev | test |
|---|---|---|---|
| bigram | 2.4548 | 2.4533 | 2.4584 |
| mlp | 2.1741 | 2.1916 | 2.1857 |

| model | T | dev NLL@T | entropy | unique | in_train | avg_len | truncated |
|---|---|---|---|---|---|---|---|
| bigram | 0.7 | 2.5216 | 2.093 | 0.80 | 0.34 | 5.26 | 0.00 |
| bigram | 1.0 | 2.4533 | 2.464 | 0.92 | 0.16 | 7.56 | 0.04 |
| mlp | 0.7 | 2.3000 | 1.699 | 0.94 | 0.34 | 5.38 | 0.00 |
| mlp | 1.0 | 2.1916 | 2.150 | 0.98 | 0.36 | 5.98 | 0.00 |

## Samples (first 6 per prompt)

- prompt='' bigram T=0.7: ya, syahaviaha, dleekaha, mannya, tryahe, chen
- prompt='' bigram T=1.0: ya, syahavilin, dleekahmangonya, tryahe, chen, ena
- prompt='' mlp T=0.7: yansy, tamilin, darek, saleighyanna, and, chel
- prompt='' mlp T=1.0: yaesyah, malin, dleek, saleinny, nelyah, anten
- prompt='a' bigram T=0.7: akeles, aly, a, a, any, a
- prompt='a' bigram T=1.0: aly, a, a, any, ayaveyi, adupolannezane
- prompt='a' mlp T=0.7: ana, any, aali, adi, adul, alenne
- prompt='a' mlp T=1.0: anoy, aashaun, adulo, ann, adzellora, alle
- prompt='ma' bigram T=0.7: mariahlamion, man, ma, ma, mali, ma
- prompt='ma' bigram T=1.0: maei, maionylyan, machaizelayane, maban, mayann, mayskako
- prompt='ma' mlp T=0.7: man, macella, mariah, mavie, marianten, mahirahmon
- prompt='ma' mlp T=1.0: maxkivi, mariag, matan, mahirahmola, maen, mainny
- prompt='jo' bigram T=0.7: jorialitan, jondrahmale, joei, jouinylyan, jochaiadan, joanan
- prompt='jo' bigram T=1.0: jonisema, jomalahe, jortorpaclilyai, jonestlsa, jon, jon
- prompt='jo' mlp T=0.7: jodia, joana, joran, jona, jon, jolsy
- prompt='jo' mlp T=1.0: jona, jon, jolskau, jowh, joey, jomily
- prompt='kay' bigram T=0.7: kayn, kay, kay, kayamalyi, kaysei, kayanaeleen
- prompt='kay' bigram T=1.0: kayn, kayjaharon, kayna, kayair, kay, kayzansuenayllea
- prompt='kay' mlp T=0.7: kaylina, kayaelee, kaylani, kaylani, kaysand, kayline
- prompt='kay' mlp T=1.0: kaylon, kayma, kayla, kaylatorpenlelya, kayna, kayse
