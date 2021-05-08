import pandas as pd
import featuretools as ft
import joblib, time, sys, os


def run():
    ember2018 = r'E:\OneDrive\OneDrive - Universiti Teknologi Malaysia (UTM)\Master Project\Workspace\dataset\ember2018'
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

    feature_matrix, feature_defs = ft.dfs(entityset=es,
                                          target_entity='metadata',
                                          n_jobs=2,
                                          max_depth=1,
                                          save_progress=r'E:\OneDrive\OneDrive - Universiti Teknologi Malaysia (UTM)\Master Project\Workspace\dataset\ember2018',
                                          )

    print(feature_matrix.shape)
    print(feature_matrix)


if __name__ == '__main__':
    run()