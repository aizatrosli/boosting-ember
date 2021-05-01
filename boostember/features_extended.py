import os,time,sys
import pandas as pd
from collections import OrderedDict

features = {
    'histogram': 256,
    'byteentropy': 256,
    'strings': 104,
    'general': 10,
    'header': 62,
    'section': 255,
    'imports': 1280,
    'exports': 128,
    'datadirectories': 30
}


class emberfeatures(object):

    def __init__(self):
        self.name = []
        self.features = self.features_()
        self.collection = self.featurescollection_()
        self.tablefeatures = self.featurestable_()

    def datadir_name_(self):
        self.datadirname = []
        datadirs = ["EXPORT_TABLE", "IMPORT_TABLE", "RESOURCE_TABLE", "EXCEPTION_TABLE", "CERTIFICATE_TABLE",
                    "BASE_RELOCATION_TABLE", "DEBUG", "ARCHITECTURE", "GLOBAL_PTR", "TLS_TABLE", "LOAD_CONFIG_TABLE",
                    "BOUND_IMPORT", "IAT", "DELAY_IMPORT_DESCRIPTOR", "CLR_RUNTIME_HEADER"]
        for i, val in enumerate(datadirs):
            if i < len(datadirs):
                self.datadirname.append(f"datadirectories_{val}_size")
                self.datadirname.append(f"datadirectories_{val}_virtual_address")
        return self.datadirname

    def featuregen_name_(self, colname, dim=None):
        if isinstance(dim, int):
            return [f'{colname}_{i}' for i in range(0, dim)]
        elif isinstance(dim, list):
            return [f'{colname}_{i}' for i in dim]
        else:
            return [colname]

    def features_(self):
        self.features = []
        self.features.extend(self.featuregen_name_('histogram', 256))

        self.features.extend(self.featuregen_name_('byteentropy', 256))

        self.features.extend(self.featuregen_name_('strings', ['numstrings', 'avlength', 'printables']) +
                             self.featuregen_name_('strings_printabledist', 96) +
                             self.featuregen_name_('strings', ['entropy', 'paths', 'urls', 'registry', 'MZ']))

        self.features.extend(self.featuregen_name_('general', ['size', 'vsize', 'has_debug', 'exports', 'imports',
                                                               'has_relocations', 'has_resources', 'has_signature',
                                                               'has_tls', 'symbols']))
        self.features.extend(self.featuregen_name_('header_coff_timestamp') +
                             self.featuregen_name_('header_coff_machine', 10) +
                             self.featuregen_name_('header_coff_characteristics', 10) +
                             self.featuregen_name_('header_optional_subsystem', 10) +
                             self.featuregen_name_('header_optional_dll_characteristics', 10) +
                             self.featuregen_name_('header_optional_magic', 10) +
                             self.featuregen_name_('header', ['optional_major_image_version',
                                                              'optional_minor_image_version',
                                                              'optional_major_linker_version',
                                                              'optional_minor_linker_version',
                                                              'optional_major_operating_system_version',
                                                              'optional_minor_operating_system_version',
                                                              'optional_major_subsystem_version',
                                                              'optional_minor_subsystem_version',
                                                              'optional_sizeof_code',
                                                              'optional_sizeof_headers',
                                                              'optional_sizeof_heap_commit']))

        self.features.extend(self.featuregen_name_('section', ['size', 'size_nonzero', 'size_empty', 'size_RX', 'size_W']) +
                             self.featuregen_name_('section_sizes_hashed', 50) +
                             self.featuregen_name_('section_entropy_hashed', 50) +
                             self.featuregen_name_('section_vsize_hashed', 50) +
                             self.featuregen_name_('section_entry_name_hashed', 50) +
                             self.featuregen_name_('section_characteristics_hashed', 50))

        self.features.extend(self.featuregen_name_('imports_libraries_hashed', 256) +
                             self.featuregen_name_('imports_hashed', 1024))

        self.features.extend(self.featuregen_name_('exports_hashed', 128))

        self.features.extend(self.datadir_name_())
        return self.features

    def featurescollection_(self):
        self.collection = OrderedDict()
        for key,val in features.items():
            fetnames = [fet for fet in self.features if key.lower() in fet.lower()]
            self.collection[key] = {
                'features': fetnames,
                'n_features': len(features),
                'major_features': fetnames[-1].split('_', 1)[0],
                'minor_features': list(set(
                    [fet.split('_', 1)[0] if len(fet.split('_')) == 2 else fet.split('_', 1)[1].rsplit('_', 1)[0] for fet in fetnames]
                )),
                'n_minor_features': len(list(set(
                    [fet.split('_', 1)[0] if len(fet.split('_')) == 2 else fet.split('_', 1)[1].rsplit('_', 1)[0] for fet in fetnames]
                ))),
            }
        return self.collection

    def featurestable_(self):
        featarr = []
        for key,val in self.collection.items():
            featdict = val
            featdict['feature'] = key
            featarr.append(featdict)
        return pd.DataFrame(featarr)
