from spmf import Spmf
import pandas as pd

class SpmfExtended(Spmf):

    def to_pandas_dataframe(self, pickle=False):
        """
        Convert output to pandas DataFrame
        pickle: Save as serialized pickle
        """
        # TODO: Optional parameter for pickle file name

        if not self.patterns_:
            self.parse_output()

        patterns_dict_list = []
        for pattern_sup in self.patterns_:
            pattern_sup = (lambda x: x)(*pattern_sup)
            parse_strings = [s if i ==0 else "#"+s for i, s in enumerate(pattern_sup.split("#"))]
            pattern = parse_strings[0]
            sup = "#"+parse_strings[1]
            sup = sup.strip()
            if not sup.startswith("#SUP"):
                print("support extraction failed")
            sup = sup.split()
            sup = sup[1]

            patterns_dict_list.append({'pattern': pattern, 'sup': int(sup)})

        df = pd.DataFrame(patterns_dict_list)
        self.df_ = df

        if pickle:
            df.to_pickle(self.output_.replace(".txt", ".pkl"))
        return df
