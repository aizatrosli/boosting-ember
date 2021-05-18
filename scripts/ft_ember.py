import pandas as pd
import featuretools as ft
import joblib, time, sys, os, gc, pydot


def run():
    ember2018 = r'/home/aizat/OneDrive/Master Project/Workspace/dataset/ember2018'
    dataset = joblib.load(os.path.join(ember2018, 'ember2018_ft_test_raw.data'))

    es = ft.EntitySet(id="ember2018")

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

    es = es.entity_from_dataframe(entity_id="header_coff",
                                  dataframe=dataset["header_coff"],
                                  index="header_coff_id")

    es = es.entity_from_dataframe(entity_id="coff_characteristics",
                                  dataframe=dataset["coff_characteristics"],
                                  index="coff_characteristics_id")

    es = es.entity_from_dataframe(entity_id="header_optional",
                                  dataframe=dataset["header_optional"],
                                  index="header_optional_id")

    es = es.entity_from_dataframe(entity_id="optional_dll_characteristics",
                                  dataframe=dataset["optional_dll_characteristics"],
                                  index="optional_dll_characteristics_id")

    es = es.entity_from_dataframe(entity_id="section",
                                  dataframe=dataset["section"],
                                  index="section_id")

    es = es.entity_from_dataframe(entity_id="section_sections",
                                  dataframe=dataset["section_sections"],
                                  index="sections_id")

    es = es.entity_from_dataframe(entity_id="sections_props",
                                  dataframe=dataset["sections_props"],
                                  index="props_id")

    es = es.entity_from_dataframe(entity_id="imports",
                                  dataframe=dataset["imports"],
                                  index="imports_id")

    es = es.entity_from_dataframe(entity_id="imports_api",
                                  dataframe=dataset["imports_api"],
                                  index="imports_api_id")

    es = es.entity_from_dataframe(entity_id="exports",
                                  dataframe=dataset["exports"],
                                  index="exports_id")

    es = es.entity_from_dataframe(entity_id="datadirectories",
                                  dataframe=dataset["datadirectories"],
                                  index="datadirectories_id")

    es = es.add_relationship(ft.Relationship(es['metadata']['sha256'],
                                             es['histogram']['sha256']))
    es = es.add_relationship(ft.Relationship(es['metadata']['sha256'],
                                             es['byteentropy']['sha256']))
    es = es.add_relationship(ft.Relationship(es['metadata']['sha256'],
                                             es['strings']['sha256']))
    es = es.add_relationship(ft.Relationship(es['metadata']['sha256'],
                                             es['general']['sha256']))

    es = es.add_relationship(ft.Relationship(es['metadata']['sha256'],
                                             es['header_coff']['sha256']))
    es = es.add_relationship(ft.Relationship(es['metadata']['sha256'],
                                             es['coff_characteristics']['sha256']))
    es = es.add_relationship(ft.Relationship(es['metadata']['sha256'],
                                             es['header_optional']['sha256']))
    es = es.add_relationship(ft.Relationship(es['metadata']['sha256'],
                                             es['optional_dll_characteristics']['sha256']))

    es = es.add_relationship(ft.Relationship(es['metadata']['sha256'],
                                             es['section']['sha256']))
    es = es.add_relationship(ft.Relationship(es['metadata']['sha256'],
                                             es['section_sections']['sha256']))
    es = es.add_relationship(ft.Relationship(es['section_sections']['sections_id'],
                                             es['sections_props']['sections_id']))

    es = es.add_relationship(ft.Relationship(es['metadata']['sha256'],
                                             es['imports']['sha256']))
    es = es.add_relationship(ft.Relationship(es['imports']['imports_id'],
                                             es['imports_api']['imports_id']))

    es = es.add_relationship(ft.Relationship(es['metadata']['sha256'],
                                             es['exports']['sha256']))
    es = es.add_relationship(ft.Relationship(es['metadata']['sha256'],
                                             es['datadirectories']['sha256']))

    print(es)
    (graph, ) = pydot.graph_from_dot_file(es.plot().save('ember2018_test.dot'))
    graph.write_svg('ember2018_test.svg')
    gc.enable()
    del dataset, graph
    gc.collect()

    features = ft.load_features(os.path.join(ember2018, 'ember2018_ft_gen5.json'))
    feature_matrix, feature_defs = ft.dfs(entityset=es,
                                          target_entity='metadata',
                                          n_jobs=1,
                                          max_depth=5,
                                          chunk_size=10000,
                                          seed_features=features,
                                          features_only=False,
                                          verbose=True,
                                          save_progress=ember2018,
                                          )

    print(feature_defs, len(feature_defs))
    print(feature_matrix.shape)
    print(feature_matrix)
    feature_matrix.to_pickle(os.path.join(ember2018, 'ember2018_ft_big.data'), compression=None)
    ft.save_features(feature_defs, os.path.join(ember2018, 'ember2018_ft_big.json'))
    


if __name__ == '__main__':
    run()
