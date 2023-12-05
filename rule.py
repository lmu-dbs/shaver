
def parse(pattern):
    return set(int(x) for x in pattern.split(","))


class Rule:
    def __init__(self, antecedent, consequent, matching_cids, support, confidence):
        self.antecedent = parse(antecedent)
        self.consequent = parse(consequent)
        self.matching_cids = matching_cids
        # self.rule_occ_per_case := {cid1: [RuleOcc1, RuleOcc2], cid2: [RuleOcc1], cid3: [RuleOcc1, RuleOcc2], ...}
        self.rule_occ_per_case = dict()
        self.support = support
        self.confidence = confidence
        self.lift = None

    def equals(self, rule):
        return self.antecedent == rule.antecedent and self.consequent == rule.consequent

    def get_elements(self):
        elements = set({})
        elements.update(self.antecedent)
        elements.update(self.consequent)
        return elements

    def __str__(self):
        return f"{self.antecedent} ==> {self.consequent}, {self.matching_cids}, {self.support}, {self.confidence}, {self.lift}"

