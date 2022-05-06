from TwitCollector import getTwits, Load_Jsons
political_groups = [['pisorgpl', 'Porozumienie__', 'SolidarnaPL'],
                    ['platforma_org', 'Nowoczesna', 'Zieloni', 'inicjatywaPL'],
                    ['KONFEDERACJA_', 'RuchNarodowy', 'Partia_KORWiN'],
                    ['__Lewica', 'partiarazem'],
                    ['nowePSL']]

for group in political_groups:
    for political_party in group:
        getTwits(political_party,False)
