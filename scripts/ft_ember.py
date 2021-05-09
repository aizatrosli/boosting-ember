import pandas as pd
import featuretools as ft
import joblib, time, sys, os, gc


def run():
    ember2018 = r'/home/aizat/OneDrive/Master Project/Workspace/dataset/ember2018'
    dataset = joblib.load(os.path.join(ember2018, 'ember2018_ft.data'))

    es = ft.EntitySet(id="ember")
    es = es.entity_from_dataframe(entity_id="metadata",
                                  dataframe=dataset['metadata'],
                                  index="sha256", )
    es = es.entity_from_dataframe(entity_id="histogram",
                                  dataframe=dataset['histogram'],
                                  index="histogram_id", )
    es = es.entity_from_dataframe(entity_id="byteentropy",
                                  dataframe=dataset["byteentropy"],
                                  index="byteentropy_id")
    es = es.entity_from_dataframe(entity_id="strings",
                                  dataframe=dataset["strings"],
                                  index="strings_id")
    es = es.entity_from_dataframe(entity_id="general",
                                  dataframe=dataset["general"],
                                  index="general_id")

    es = es.add_relationship(ft.Relationship(es['metadata']['sha256'],
                                             es['histogram']['sha256']))
    es = es.add_relationship(ft.Relationship(es['metadata']['sha256'],
                                             es['byteentropy']['sha256']))
    es = es.add_relationship(ft.Relationship(es['metadata']['sha256'],
                                             es['strings']['sha256']))
    es = es.add_relationship(ft.Relationship(es['metadata']['sha256'],
                                             es['general']['sha256']))

    print(es)
    gc.enable()
    del dataset
    gc.collect()

    feature_matrix, feature_defs = ft.dfs(entityset=es,
                                          target_entity='metadata',
                                          n_jobs=5,
                                          max_depth=3,
                                          chunk_size=1000,
                                          features_only=False,
                                          verbose=True,
                                          save_progress=ember2018,
                                          )

    print(feature_defs, len(feature_defs))
    print(feature_matrix.shape)
    print(feature_matrix)
    joblib.dump(feature_matrix, os.path.join(ember2018, 'ember2018_ft_gen5.data'))
    ft.save_features(feature_defs, os.path.join(ember2018, 'ember2018_ft_gen5.json'))
    


if __name__ == '__main__':
    run()
