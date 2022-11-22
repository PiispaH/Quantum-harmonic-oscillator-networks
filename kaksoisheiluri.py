import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import time
#import matplotlib;matplotlib.use("TkAgg")
from matplotlib.animation import PillowWriter


def f0(theta_1_prime):
    """
    Differentiaaliyhtälö :math:`\\theta_1' = \\omega_1`

    Args:
        theta_1_prime (float): ensimmäisen heilurin kulman aikaderivaatta

    Returns:
        float: Ensimmäisen heilurin kulmanopeus
    """

    omega_1 = theta_1_prime

    return omega_1


def f1(theta_2_prime):
    """
    Differentiaaliyhtälö :math:`\\theta_2' = \\omega_2`

    Args:
        theta_2_prime (float): Toisen heilurin kulman aikaderivaatta

    Returns:
        float: Toisen heilurin kulmanopeus
    """

    omega_2 = theta_2_prime

    return omega_2


def f2(omega_1, omega_2, theta_1, theta_2, g, m_1, m_2, L_1, L_2):
    """
    Differentiaaliyhtälö ensimmäisen heilurin kulmanopeuden derivaatalle

    Args:
        omega_1 (float): Ensimmäisen helurin kulmanopeus
        omega_2 (float): Toisen heilurin kulmanopeus
        theta_1 (float): Ensimmäisen helurin kulma
        theta_2 (float): Toisen helurin kulma
        g (float): Putoamiskiihtyvyys
        m_1 (float): Ensimmäisen kuulan massa
        m_2 (float): Toisen kuulan massa
        L_1 (float): Ensimmäisen tangon pituus
        L_2 (float): Toisen tangon pituus

    Returns:
        float: omega_1_prime
    """

    # Lyhennys kulmien erotukselle
    delta = theta_1 - theta_2

    # Itse differentiaaliyhtälö
    omega_1_prime = (m_2 * g * np.sin(theta_2) * np.cos(delta) - m_2 * np.sin(delta) * (
            L_1 * omega_1 ** 2 * np.cos(delta) + L_2 * omega_2 ** 2) - (m_1 + m_2) * g * np.sin(theta_1)) / (
                            L_1 * (m_1 + m_2 * (np.sin(delta)) ** 2))

    return omega_1_prime


def f3(omega_1, omega_2, theta_1, theta_2, g, m_1, m_2, L_1, L_2):
    """
    Differentiaaliyhtälö toisen heilurin kulmanopeuden derivaatalle

    Args:
        omega_1 (float): Ensimmäisen helurin kulmanopeus
        omega_2 (float): Toisen heilurin kulmanopeus
        theta_1 (float): Ensimmäisen helurin kulma
        theta_2 (float): Toisen helurin kulma
        g (float): Putoamiskiihtyvyys
        m_1 (float): Ensimmäisen kuulan massa
        m_2 (float): Toisen kuulan massa
        L_1 (float): Ensimmäisen tangon pituus
        L_2 (float): Toisen tangon pituus

    Returns:
        float: omega_2_prime
    """

    # Lyhennys kulmien erotukselle
    delta = theta_1 - theta_2

    # Itse differentiaaliyhtälö
    omega_2_prime = ((m_1 + m_2) * (L_1 * omega_1 ** 2 * np.sin(delta) - g * np.sin(theta_2) + g * np.sin(theta_1) * np.cos(
            delta)) + m_2 * L_2 * omega_2 ** 2 * np.sin(delta) * np.cos(delta)) / (L_2 * (m_1 + m_2 * (np.sin(delta)) ** 2))

    return omega_2_prime


def runge_kutta(h, askeleet, yhtalot, omega_1, omega_2, theta_1, theta_2, g, m_1, m_2, L_1, L_2):
    """
    Ratkaisee heilureiden liikeyhtälöt pienin aikavälein neljännen kertaluvun Runge-Kutta -menetelmällä.
    Iteraatioaskeleita suoritetaan parametreissa annettu määrä.

    Args:
        h (float): Askelpituus
        askeleet (int): Askelten määrä
        yhtalot (list): Kaikki differentiaaliyhtälöt
        omega_1 (float): Ensimmäisen helurin kulmanopeus
        omega_2 (float): Toisen heilurin kulmanopeus
        theta_1 (float): Ensimmäisen helurin kulma
        theta_2 (float): Toisen helurin kulma
        g (float): Putoamiskiihtyvyys
        m_1 (float): Ensimmäisen kuulan massa
        m_2 (float): Toisen kuulan massa
        L_1 (float): Ensimmäisen tangon pituus
        L_2 (float): Toisen tangon pituus

    Returns:
        ndarray, ndarray, ndarray, ndarray, ndarray: Kuulien x-koordinaatit kaikilla ajanhetkillä,
        Kuulien y-koordinaatit kaikilla ajanhetkillä,
        Kaikki ajanhetket,
        Systeemin potentiaalienergiat kaikilla ajanhetkillä,
        Systeemin liike-energiat kaikilla ajanhetkillä
    """

    # Määritetään tyhjät listat kuulien x- ja y-koordinaateille
    kuulat_x = []
    kuulat_y = []

    # Määritetääb lista heilurien kulmille
    kulmat = []

    # Alustetaan listat liike- ja potentiaalienergioille sekä ajanhetkille
    liike_energiat = []
    potentiaali_energiat = []
    ajat = []

    # Iteroidaan halutun askelmäärän verran
    for i in range(askeleet):

        # Lasketaan K1 kertoimet
        k1_0 = yhtalot[0](omega_1)  # theta_1'
        k1_1 = yhtalot[1](omega_2)  # theta_2'
        k1_2 = yhtalot[2](omega_1, omega_2, theta_1, theta_2, g, m_1, m_2, L_1, L_2)  # omega_1'
        k1_3 = yhtalot[3](omega_1, omega_2, theta_1, theta_2, g, m_1, m_2, L_1, L_2)  # omega_2'

        # Lasketaan K2 kertoimet
        k2_0 = yhtalot[0](omega_1 + h/2 * k1_2)
        k2_1 = yhtalot[1](omega_2 + h/2 * k1_3)
        k2_2 = yhtalot[2](omega_1, omega_2, theta_1 + h / 2 * k1_0, theta_2, g, m_1, m_2, L_1, L_2)
        k2_3 = yhtalot[3](omega_1, omega_2, theta_1, theta_2 + h / 2 * k1_1, g, m_1, m_2, L_1, L_2)

        # Lasketaan K3 kertoimet
        k3_0 = yhtalot[0](omega_1 + h/2 * k2_2)
        k3_1 = yhtalot[1](omega_2 + h/2 * k2_3)
        k3_2 = yhtalot[2](omega_1, omega_2, theta_1 + h/2 * k2_0, theta_2, g, m_1, m_2, L_1, L_2)
        k3_3 = yhtalot[3](omega_1, omega_2, theta_1, theta_2 + h/2 * k2_1, g, m_1, m_2, L_1, L_2)

        # Lasketaan K4 kertoimet
        k4_0 = yhtalot[0](omega_1 + h * k3_2)
        k4_1 = yhtalot[1](omega_2 + h * k3_3)
        k4_2 = yhtalot[2](omega_1, omega_2, theta_1 + h * k3_0, theta_2, g, m_1, m_2, L_1, L_2)
        k4_3 = yhtalot[3](omega_1, omega_2, theta_1, theta_2 + h * k3_1, g, m_1, m_2, L_1, L_2)

        # Päivitetään uudet kulmat ja kulmanopeudet määritettyjen kertoimien avulla
        theta_1 += h/6 * (k1_0 + 2*k2_0 + 2*k3_0 + k4_0)
        theta_2 += h/6 * (k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
        omega_1 += h/6 * (k1_2 + 2*k2_2 + 2*k3_2 + k4_2)
        omega_2 += h/6 * (k1_3 + 2*k2_3 + 2*k3_3 + k4_3)

        # Tallennetaan kulmat
        kulmat.append([theta_1, theta_2])

        # Lasketaan ja lisätään listaan uudet kuulien paikat
        x_1, y_1, x_2, y_2 = paikkojen_maaritys(theta_1, theta_2, L_1, L_2)
        kuulat_x.append([x_1, x_2])
        kuulat_y.append([y_1, y_2])

        # Määritetään potentiaali- ja liike-energiat
        U = potentiaali_energia(theta_1, theta_2, g, m_1, m_2, L_1, L_2)
        potentiaali_energiat.append(U)

        K = liike_energia(theta_1, theta_2, omega_1, omega_2, m_1, m_2, L_1, L_2)
        liike_energiat.append(K)

        # Tallennetaan ajanhetki
        ajat.append(i * h)

    # Muutetaan koordinaatit, ajanhetket, energiat ja kulmat array -muotoon
    kuulat_x = np.array(kuulat_x)
    kuulat_y = np.array(kuulat_y)
    kulmat = np.array(kulmat)
    ajat = np.array(ajat)
    potentiaali_energiat = np.array(potentiaali_energiat)
    liike_energiat = np.array(liike_energiat)

    return kuulat_x, kuulat_y, ajat, potentiaali_energiat, liike_energiat, kulmat


def potentiaali_energia(theta_1, theta_2, g, m_1, m_2, L_1, L_2):
    # Määrittää potentiaalienergian tietyllä ajanhetkellä

    # Määritetään ensin potentiaalienergia siten, että nollataso on kuulien puolessavälissä
    U = -(m_1 + m_2) * L_1 * g * np.cos(theta_1) - m_2 * L_2 * g * np.cos(theta_2) + 0.5 * g * (m_1 + m_2)

    return U


def liike_energia(theta_1, theta_2, omega_1, omega_2, m_1, m_2, L_1, L_2):
    # Määrittää liike-energian tietyllä ajanhetkellä

    K = 0.5 * m_1 * L_1 ** 2 * omega_1 ** 2 + 0.5 * m_2 * (
            L_1 ** 2 * omega_1 ** 2 + L_2 ** 2 * omega_2 ** 2 + 2 * L_1 * L_2 * omega_1 * omega_2 * np.cos(
        theta_1 - theta_2))

    return K


def piirto(frame, kuulat_x, kuulat_y):
    # Osia funktion koodista otettu kurssin luennon 5 ohjelmasta A5_fem.py funktiosta draw

    # Luodaan akselit
    plt.clf()
    ax = plt.axes()

    # Asetetaan akselien rajoiksi 0 ja 1
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')

    # Piirtää kuulat oikeille paikoille, ensimmäinen kuula pysyy aina paikallaan keskellä kuvaa
    plt.plot((0.5, kuulat_x[frame][0]), (0.5, kuulat_y[frame][0]), 'ko', linestyle="-",
             ms=12, lw=2.5)
    plt.plot((kuulat_x[frame][0], kuulat_x[frame][1]), (kuulat_y[frame][0], kuulat_y[frame][1]), 'ko', linestyle="-",
             ms=12, lw=2.5)


def animaatio(kuulat_x, kuulat_y, tallennus):
    # Osia funktion koodista otettu kurssin luennon 5 ohjelmasta A5_fem.py funktiosta animate

    # Animoi heilurin
    fig = plt.figure()
    anim = ani.FuncAnimation(fig, piirto, len(kuulat_x), fargs=(kuulat_x, kuulat_y), interval=1)

    # Jos halutaan, tallennetaan animaatio gifinä
    if tallennus:
        writer = PillowWriter(fps=30)
        anim.save("Heiluri.gif", writer=writer)

    plt.show()


def paikkojen_maaritys(theta_1, theta_2, L_1, L_2):
    # Määrittää kuulien paikat kulmien avulla

    # Laskee molempien kuulien x-koordinaatit trigonometrian avulla
    x_1 = 0.5 + L_1 * np.sin(theta_1)
    x_2 = x_1 + L_2 * np.sin(theta_2)

    # Laskee molempien kuulien y-koordinaatit trigonometrian avulla
    y_1 = 0.5 - L_1 * np.cos(theta_1)
    y_2 = y_1 - L_2 * np.cos(theta_2)

    # Palautetaan koordinaatit
    return x_1, y_1, x_2, y_2


def energia_plottaus(t, U, K):
    # Funktion koodi on kopioitu luennon 4 ohjelmasta A5_2dmd.py funktiosta main

    ax = plt.axes()

    # Plotataan liike-, potentiaali- ja kokonaisenergia
    plt.plot(t, K, label="Liike-energia")
    plt.plot(t, U, label="Potentiaali-energia")
    plt.plot(t, U + K, label="Kokonais-energia")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("E")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    plt.show()



def data_monta_heiluria(U_keskiarvot, K_keskiarvot, plotataanko, N):
    # Analysoi monen heilurin simulaatioista kerättyä dataa

    # Jaetaan potentiaalienergian keskiarvot kahteen eri listaan, siten että
    # toisessa on parilliset indeksit ja toisessa parittomat
    U_keskiarvot_1 = U_keskiarvot[::2]
    U_keskiarvot_2 = U_keskiarvot[1::2]

    # Jos N on suurempi kuin yksi ja pariton, jotta dataa olisi helppo käsitellä,
    # poistetaan ensimmäisestä arraysta viimeinen alkio, jotta arrayt ovat yhtä pitkiä
    if N % 2 != 0:
        U_keskiarvot_1 = np.delete(U_keskiarvot_1, -1)

    # Luodaan array joka sisältää molemmat energia -arrayt
    data = np.array([U_keskiarvot_1, U_keskiarvot_2])

    # Lasketaan potentiaalienergioiden keskiarvojen korrelaatiokerroin
    cov = np.corrcoef(data)[0][1]
    print(cov)

    # Jos data halutaan plotata
    if plotataanko:
        ax = plt.axes()

        # Luodaaan lista, jossa on alkiot [1, 2, 3,...,N], jossa N on heilureiden lukumäärä
        simulaatiot = list(range(1, N + 1))

        # Plotataan U:n keskiarvot heilurin järjestysnumeron funktiona
        plt.plot(simulaatiot, U_keskiarvot, label="Potentiaalienergian keskiarvot")
        plt.legend()
        plt.xlabel("t")
        plt.ylabel("⟨U⟩")

        ax.set_xlim(left=1)
        plt.show()


def simulaatio(h, askeleet, muutos, g, m_1, m_2, L_1, L_2, theta_1, theta_2,
               omega_1, omega_2, N, tulostus, animointi, plotti, tallennus, nopeus):
    # Suorittaa koko simuloinnin

    # Tallennetaan diff. yhtälöt listaan ja alustetaan listat energioita varten
    yhtalot = [f0, f1, f2, f3]
    U_keskiarvot = []
    K_keskiarvot = []

    # Jos simuloidaan vain yksi heiluri
    if N == 1:

        # Aloitetaan ajanotto jotta voidaan määrittää simuloimiseen kestänyt aika
        alkoi = time.time()

        # Ratkaistaan heilurin liike ja energiat Runge-Kuttan avulla
        kuulat_x, kuulat_y, ajat, potentiaali_energiat, liike_energiat, kaikki_kulmat = runge_kutta(h, askeleet, yhtalot,
                                                                                    omega_1, omega_2, theta_1,
                                                                                    theta_2, g, m_1, m_2, L_1, L_2)
        # Lopetetaan ajanotto
        loppui = time.time()

        # Jos haluaan animoida
        if animointi:

            # Jotta animaatio liikkuisi järkevällä nopeudella, ohitetaan osa frameista.
            # Osuus frameista jotka ohitetaan määritetään parametrin nopeus avulla
            kuulat_x = kuulat_x[0::nopeus]
            kuulat_y = kuulat_y[0::nopeus]

            # Animoidaan heilurin liike ja tallennetaan se jos halutaan
            animaatio(kuulat_x, kuulat_y, tallennus)

        # Jos halutaan plotata
        if plotti:

            # Plotataan liike-, potentiaali ja kokonaisenergia
            energia_plottaus(ajat, potentiaali_energiat, liike_energiat)

        # Määritetään kokonaisenergia alussa ja lopussa sekä määritetään näiden avulla
        # Runge-Kuttan virheestä johtuva kokonaisenergian prosentuaalinen muutos
        E_tot_alussa = potentiaali_energiat[0] + liike_energiat[0]
        E_tot_lopussa = potentiaali_energiat[-1] + liike_energiat[-1]
        E_muutos = 100 * (E_tot_lopussa - E_tot_alussa) / E_tot_alussa

        # Lisätään lasketut energioiden keskiarvot palautettaviin listoihin
        U_keskiarvot.append(np.average(potentiaali_energiat))
        K_keskiarvot.append(np.average(liike_energiat))

        # Jos halutaan tulostaa
        if tulostus:
            print(f"\nSimuloimiseen kului {(loppui - alkoi):.2f} sekuntia")
            print(f"Kokonaisenergian suhteellinen muutos: {E_muutos:.2f} %")
            print(f"\nPotentiaalienergian keskiarvo: {U_keskiarvot[0]:.3f}")
            print(f"Liike-energian keskiarvo: {K_keskiarvot[0]:.3f}")

    else:
        # Jos halutaan suorittaa monta simulaatiota kerralla ja analysoida dataa

        # Aloitetaan ajanotto
        alkoi = time.time()

        # Alustetaan listat pyöräyttämällä ensimmäinen kierros luupin ulkopuolella
        kuulat_x, kuulat_y, ajat, potentiaali_energiat, liike_energiat, kulmat1 = runge_kutta(h, askeleet, yhtalot,
                                                                                     omega_1, omega_2,
                                                                                     theta_1, theta_2, g, m_1, m_2,
                                                                                     L_1, L_2)

        # Alustetaan listat kuulien paikoille ja systeemin energioille sekä näiden keskiarvoille
        # ensimmäisen heilurin datalla
        kuulat_x = [kuulat_x]
        kuulat_y = [kuulat_y]
        potentiaali_energiat = [potentiaali_energiat]
        liike_energiat = [liike_energiat]
        U_keskiarvot.append(np.average(potentiaali_energiat))
        K_keskiarvot.append(np.average(liike_energiat))


        # Luodaan lista jossa ensimmäisenä alkiona on ensimmäisen heilurin kulmat
        kaikki_kulmat = [kulmat1]

        # Suoritetaan loput kierrokset ja lisätään tulokset listoihin
        for i in range(1, N):

            # Lisätään pieni kulman muutos ennen uuden heilurin suureiden ratkaisemista
            theta_2 += muutos

            # Määritetään uuden heilurin suureet Runge-Kuttan avulla
            x, y, ajat, potentiaali_energia, liike_energia, kulmat = runge_kutta(h, askeleet, yhtalot,
                                                                                         omega_1, omega_2,
                                                                                         theta_1, theta_2, g, m_1, m_2,
                                                                                         L_1, L_2)
            # Lisätään listoihin uutta heiluria vastaavat suureet
            kuulat_x.append(x)
            kuulat_y.append(y)

            potentiaali_energiat.append(potentiaali_energia)
            liike_energiat.append(liike_energia)

            U_keskiarvot.append(np.average(potentiaali_energia))
            K_keskiarvot.append(np.average(potentiaali_energia))

            # Jos heilureita on kaksi
            if N == 2:
                # Lisätään kulmalistaan myös toisen heilurin kulmat ja muutetaan lista arrayksi
                kaikki_kulmat.append(kulmat)
                kaikki_kulmat = np.array(kaikki_kulmat)

        # Kun kaikki heilurit on simuloitu, lopetetaan ajanotto
        loppui = time.time()

        # Muutetaan energioiden keskiarvot array -muotoon
        U_keskiarvot, K_keskiarvot = np.array(U_keskiarvot), np.array(K_keskiarvot)

        # Jos halutaan tulostaa
        if tulostus:
            print(f"\nSimuloimiseen kului {(loppui - alkoi):.2f} sekuntia")
            print("Simuloitiin", N, "heiluria.")

            print("\nPotentiaalienergiat:")

            # Tulostetaan kaikki potentiaalienergioiden keskiarvot
            for i in U_keskiarvot:
                print(f"{i:.4f}")

    # Palautetaan energioiden keskiarvot, yhden tai kahden heilurin kulmat
    # sekä ajanhetket jolloin kaikki heilureiden suureet on ratkaistu
    return U_keskiarvot, K_keskiarvot, ajat, kaikki_kulmat


def kulma_plottaus(t, kulmat, N):
    # Alustetaan akselit
    ax = plt.axes()

    # Jos heilureita on kaksi
    if N == 2:
        # Tallennetaan eri heilureiden ensimmäisen tangon kulmat listoihin
        kulmat1 = [i[0] for i in kulmat[1]]
        kulmat2 = [i[0] for i in kulmat[0]]

        # Plotataan heilureiden kulmien sinit eri värisinä samassa kuvassa
        plt.plot(t, np.sin(kulmat1), c="k", label="Heiluri 1, θ_1")
        plt.plot(t, np.sin(kulmat2), c="r", label="Heiluri 2, θ_1")
        plt.legend(loc=1)

    else:
        # Plotataan yhden heilurin molempien kulmien sinit
        plt.plot(t, kulmat)

    # Asetetaan nimet akseleille ja siistitään kuvaa hieman
    plt.xlabel("t")
    plt.ylabel("θ")
    ax.set_xlim(left=0)
    plt.show()


def main():
    """ Muutettavat suureet alkavat """

    #Simulaatio:
    # Kuinka monta simulaatiota suoritetaan yhtäaikaisesti
    N = 1

    # Simulaation kesto
    t = 15

    # Simulaation nopeus, käytännössä kuinka monta kuvaa hypätään yli kun
    # heiluria animoidaan, kun t=15 ja h=0.0001, nopeuden on hyvä olla noin 90.
    # Jos
    nopeus = 90

    #Runge-Kutta:
    # Askelpituus
    h = 0.0001

    #Heiluriin liittyvät vakiot:
    # putoamiskiihtyvyys
    g = 9.81

    # Ylemmän (1) ja alemman (1) kuulan massat
    m_1 = 1
    m_2 = 5

    # Ylemmän ja alemman tangon pituudet
    L_1 = 0.3
    L_2 = 0.1

    #Heiluri:
    # Tankojen kulmat alussa, nollakulma osoittaa molemmilla alaspäin
    # ja positiivinen kiertosuunta on vastapäivään
    theta_1 = np.pi / 2
    theta_2 = np.pi / 2

    # Heilureiden kulmanopeudet alussa
    omega_1 = 0
    omega_2 = 0

    # Jos simuloidaan monta heiluria samalla:
    # Kuinka paljon toisen tangon alkukulmaa muutetaan simulaatioiden
    # välissä (rad)
    muutos = 0.0001

    #Data:
    # Tehdäänkö tulosteet
    tulostetaanko = True

    # Jos simuloidaan vain yksi heluri, voidaan päättää
    # tehdäänkö animointia tai plottauksia
    plotataanko = True
    animoidaanko = False

    # Jos animoidaan, tallennetaanko animaatio
    tallennetaanko = False

    """ Muutettavat suureet loppuvat """

    # Askelten määrä
    askeleet = int(t / h)

    # Pyöräytetään simulaatio
    U_keskiarvot, K_keskiarvot, ajat, kulmat = simulaatio(h, askeleet, muutos, g, m_1, m_2, L_1, L_2, theta_1, theta_2, omega_1,
               omega_2, N, tulostetaanko, animoidaanko, plotataanko, tallennetaanko, nopeus)

    # Jos heilureita on yksi tai kaksi ja halutaan tehdä plottaukset
    if plotataanko and N < 3:

        # Jos kyseessä yksi heiluri, plotataan heilurin molempien kulmien sinit,
        # muuten plotataan kahden eri heilurin ensimmäisen tangon kulmien sinit
        kulma_plottaus(ajat, kulmat, N)

    # Jos heilureita on enemmän kuin yksi
    if N > 1:
        # Kerätään dataa monesta heilurista
        data_monta_heiluria(U_keskiarvot, K_keskiarvot, plotataanko, N)


if __name__ == '__main__':
    main()
