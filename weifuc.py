from waifuc.action import HeadCountAction, AlignMinSizeAction, ModeConvertAction, NoMonochromeAction, ClassFilterAction, FilterSimilarAction
from waifuc.export import SaveExporter
from waifuc.source import DanbooruSource, GcharAutoSource, Rule34Source, E926Source, GelbooruSource, SafebooruOrgSource, SafebooruSource, SankakuSource
# see https://github.com/deepghs/waifuc

def main_method(method, text_list, output):
    #source = GcharAutoSource('peni_parker')
    #source = DanbooruSource(['scylla_(hades)'])
    source = method(text_list)
    
    source.attach(
        # preprocess images with white background RGB
        ModeConvertAction('RGB', 'white'),

        # pre-filtering for images
        #NoMonochromeAction(),  # no monochrome, greyscale or sketch
        #ClassFilterAction(['illustration', 'bangumi']),  # no comic or 3d
        # RatingFilterAction(['safe', 'r15']),  # filter images with rating, like safe, r15, r18
        FilterSimilarAction('all'),  # filter duplicated images

        # only 1 head,
        #HeadCountAction(1),

        # if shorter side is over 640, just resize it to 640
        AlignMinSizeAction(640),
    )[:50].export(  # only first 10 images
        # save images (with meta information from danbooru site)
        SaveExporter(output)
    )



if __name__ == '__main__':
    name_list = [#'asterius','athena','cerberus','chaos','charon','chronos'
        #'demeter','dionysus','dora','eris','eurydice','hecate','hephaestus','hera','heracles','hestia','hydra','hypnos'
        #'dusa','moros','nemesis','odyssus','orpheus','patroclus','persephone','polyphemus','poseidon','scylla','selene','sisyphus','skelly','theseus','tisiphone','zeus'
        #'gwenom', 'storm_(x-men)', 'sabretooth', 'azazel_(x-men)', 'elektra_natchios', 'x-23'
        #'negasonic_teenage_warhead', 'domino_(marvel)', 'yukio_(x-men)', 'wolverine_(x-men)', 'angel_dust_(x-men)', 'colossus', 'juggernaut', 'morena_baccarin', 'wade_wilson'
        #'renamon', 'angewoman', 'fairimon', 'ladydevimon', 'beelstarmon', 'sakuyamon', 'gatomon', 'lilithmon', 'wargreymon', 'rosemon', 'ranamon', 'venusmon', 'guilmon', 'shutumon', 'mervamon', 'bastemon', 
        'angewomon', 'gabumon', 'biyomon',
                 ]

    for name in name_list:
        text_list = [f'{name}']
        output = f'data/digimon/{name}'
        main_method(DanbooruSource, text_list,output)
        #main_method(GcharAutoSource, text_list,output)
        main_method(Rule34Source, text_list,output)
        main_method(E926Source, text_list,output) #for animal
        #main_method(GelbooruSource, text_list,output) #very similar to danbooru
    #main_method(SafebooruOrgSource, text_list,output) #
    #main_method(SafebooruSource, text_list,output) #
    #main_method(SankakuSource, text_list,output) #