class TaxonomyClass:
    # For HOGraspNet Label
    def __init__(self):
        self.map = {
            'power': [1, 2, 3, 4, 5, 10, 11, 17, 18, 19, 22, 26, 28, 30, 31],
            'intermediate': [16, 20, 23, 25, 29],
            'precision': [7, 9, 12, 13, 14, 24, 27, 33]
        }
        self.colors = {
            'power': '#d95f02',
            'intermediate': '#7570b3',
            'precision': '#1b9e77',
            'unknown': '#8c8c8c',
        }
        '''
        self.label_to_category = {
            1: 'power',
            2: 'power',
            ...
            27: 'precision',
            ...
        '''
        self.label_to_category = {
            int(label): category
            for category, labels in self.map.items()
            for label in labels
        }

        self.category_to_integer = {
            'power': 0,
            'intermediate': 1,
            'precision': 2,
        }

        self.label_to_category_integer = {
            label: self.category_to_integer[category]
            for label, category in self.label_to_category.items()
        }

    def category_for_label(self, label):
        return self.label_to_category.get(int(label), 'unknown')

    def color_for_label(self, label):
        return self.colors[self.category_for_label(label)]

    def colors_for_labels(self, labels):
        return [self.color_for_label(label) for label in labels]