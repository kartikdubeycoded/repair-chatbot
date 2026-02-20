"""Retrieval validation and session metrics."""


class RetrievalValidator:
    """Validates retrieval quality and tracks runtime metrics."""

    def __init__(self):
        self.session_metrics = {
            'total_queries': 0,
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0,
            'no_results': 0,
            'average_accuracy': 0.0,
            'hallucination_count': 0,
        }

    def log_retrieval_performance(self, query: str, search_results: dict, validation: dict):
        """Print per-query performance and update session counters."""
        confidence_level = search_results.get('confidence_level', 'none')
        stage_used = search_results.get('stage_used')
        results = search_results.get('results', [])

        print("\n" + "=" * 60)
        print("RETRIEVAL METRICS")
        print("=" * 60)
        print(f'Query: "{query}"')
        if stage_used:
            print(f"Confidence: {confidence_level.upper()} (Stage {stage_used})")
        else:
            print(f"Confidence: {confidence_level.upper()}")

        print(f"Results: {len(results)} guides found")
        if results:
            top_result = results[0]
            print(f"Top Match: \"{top_result['metadata'].get('title', 'Unknown')}\" (score: {top_result['score']})")

        print(f"Accuracy: {validation.get('accuracy_score', 0):.0f}/100")
        print(f"Hallucination Risk: {validation.get('hallucination_risk', 'unknown').upper()}")

        issues = validation.get('issues_detected', [])
        if issues:
            print(f"Issues: {', '.join(issues)}")
        print("=" * 60 + "\n")

        self.session_metrics['total_queries'] += 1

        if confidence_level == 'high':
            self.session_metrics['high_confidence'] += 1
        elif confidence_level == 'medium':
            self.session_metrics['medium_confidence'] += 1
        elif confidence_level == 'low':
            self.session_metrics['low_confidence'] += 1
        else:
            self.session_metrics['no_results'] += 1

        total = self.session_metrics['total_queries']
        current_avg = self.session_metrics['average_accuracy']
        accuracy = validation.get('accuracy_score', 0.0)
        self.session_metrics['average_accuracy'] = ((current_avg * (total - 1)) + accuracy) / total

        if validation.get('hallucination_risk') == 'high':
            self.session_metrics['hallucination_count'] += 1

    def print_session_summary(self):
        """Print session summary metrics."""
        metrics = self.session_metrics
        total = metrics['total_queries']
        if total == 0:
            return

        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"Total Queries: {total}")
        print(f"High Confidence: {metrics['high_confidence']} ({metrics['high_confidence'] / total * 100:.1f}%)")
        print(f"Medium Confidence: {metrics['medium_confidence']} ({metrics['medium_confidence'] / total * 100:.1f}%)")
        print(f"Low Confidence: {metrics['low_confidence']} ({metrics['low_confidence'] / total * 100:.1f}%)")
        print(f"No Results: {metrics['no_results']} ({metrics['no_results'] / total * 100:.1f}%)")
        print(f"Average Accuracy: {metrics['average_accuracy']:.1f}/100")
        print(f"Hallucinations Detected: {metrics['hallucination_count']}")
        print("=" * 60 + "\n")


validator = RetrievalValidator()
