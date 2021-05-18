import os,sys,gc,itertools,tqdm

sys.path.append(os.path.dirname(os.getcwd()))
from boostember import *

config = {
    'booster': ['lgb'],
    'experiment': ['emberboosting'],
    'n_estimator': [10, 50, 100, 500, 1000],
    'defaultdataset': [True],
}

#farr=['MEAN(coff_characteristics.coff_characteristics_hash_13)', 'MAX(datadirectories.size_CERTIFICATE_TABLE)', 'MEAN(imports_api.imports.imports_hash_20)', 'MAX(header_optional.optional_major_subsystem_version)', 'MIN(section_sections.MEAN(sections_props.props_hash_14))', 'MEAN(imports_api.imports_api_hash_10)', 'MAX(datadirectories.size_RESOURCE_TABLE)', 'STD(imports.imports_hash_16)', 'MODE(header_optional.optional_subsystem) = WINDOWS_GUI']
farr=['header_coff_characteristics_3', 'imports_libraries_hashed_117', 'header_optional_major_subsystem_version', 'strings_printabledist_74', 'header_optional_subsystem_8', 'datadirectories_RESOURCE_TABLE_size', 'datadirectories_CERTIFICATE_TABLE_size', 'datadirectories_DEBUG_virtual_address', 'header_coff_characteristics_0']
fstr = ' and featureselect' if farr is not None else ''

def run():

    keys, values = zip(*config.items())
    experiments = tqdm.tqdm([dict(zip(keys, v)) for v in itertools.product(*values)])

    for experimentdict in experiments:
        exp = f'no AFE without featurehaser{fstr}' if experimentdict['defaultdataset'] else f'AFE without featurehaser{fstr}'
        experiments.set_description("{} \n".format("\t".join(f"[{k}]: {v}" for k, v in experimentdict.items())))
        run = Boosting(f'{experimentdict["booster"]} n{experimentdict["n_estimator"]} {exp}',
                       experiment=experimentdict['experiment'], booster=experimentdict["booster"],
                       n_estimator=experimentdict["n_estimator"], defaultdataset=experimentdict["defaultdataset"],
                       features=farr,
                       dataset='/home/aizat/OneDrive/Master Project/Workspace/dataset/ember2018', n_jobs=21, verbose=False)
        run.main(cv=True, n=5)
        del run
        gc.collect()


if __name__ == '__main__':
    run()

