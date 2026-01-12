"""
Report generation module for safety analytics
"""

import csv
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate safety compliance reports in various formats
    """
    
    def __init__(self, output_dir: str = "outputs/reports"):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Report generator initialized: {output_dir}")
    
    def generate_daily_report(
        self,
        date: datetime,
        violations: List[Dict],
        stats: Dict,
        format: str = "csv"
    ) -> Optional[str]:
        """
        Generate daily safety report
        
        Args:
            date: Report date
            violations: List of violations
            stats: Statistics dictionary
            format: Output format ('csv', 'pdf', 'both')
            
        Returns:
            Path to generated report(s)
        """
        date_str = date.strftime('%Y-%m-%d')
        
        if format == 'csv' or format == 'both':
            csv_path = self._generate_csv_report(date_str, violations, stats)
        
        if format == 'pdf' or format == 'both':
            pdf_path = self._generate_pdf_report(date_str, violations, stats)
        
        if format == 'both':
            return f"CSV: {csv_path}, PDF: {pdf_path}"
        elif format == 'pdf':
            return pdf_path
        else:
            return csv_path
    
    def _generate_csv_report(
        self,
        date_str: str,
        violations: List[Dict],
        stats: Dict
    ) -> str:
        """
        Generate CSV report
        
        Args:
            date_str: Date string
            violations: List of violations
            stats: Statistics dictionary
            
        Returns:
            Path to CSV file
        """
        filename = f"safety_report_{date_str}.csv"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow(['Helmet Detection Safety Report'])
                writer.writerow(['Date', date_str])
                writer.writerow([])
                
                # Write summary statistics
                writer.writerow(['SUMMARY STATISTICS'])
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Total Violations', stats.get('total_violations', 0)])
                writer.writerow(['Helmet Compliant', stats.get('helmet_count', 0)])
                writer.writerow(['No Helmet', stats.get('no_helmet_count', 0)])
                writer.writerow(['Compliance Rate', f"{stats.get('compliance_rate', 0) * 100:.2f}%"])
                writer.writerow([])
                
                # Write violations detail
                writer.writerow(['VIOLATIONS DETAIL'])
                writer.writerow(['Timestamp', 'Camera ID', 'Type', 'Confidence', 'Severity', 'Snapshot'])
                
                for violation in violations:
                    writer.writerow([
                        violation.get('timestamp', ''),
                        violation.get('camera_id', ''),
                        violation.get('violation_type', violation.get('type', '')),
                        f"{violation.get('confidence', 0):.2f}",
                        violation.get('severity', 'N/A'),
                        violation.get('snapshot_path', 'N/A')
                    ])
            
            logger.info(f"CSV report generated: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error generating CSV report: {e}")
            return ""
    
    def _generate_pdf_report(
        self,
        date_str: str,
        violations: List[Dict],
        stats: Dict
    ) -> str:
        """
        Generate PDF report
        
        Args:
            date_str: Date string
            violations: List of violations
            stats: Statistics dictionary
            
        Returns:
            Path to PDF file
        """
        filename = f"safety_report_{date_str}.pdf"
        filepath = self.output_dir / filename
        
        try:
            doc = SimpleDocTemplate(str(filepath), pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1f4788'),
                spaceAfter=30,
                alignment=1  # Center
            )
            title = Paragraph("Safety Helmet Compliance Report", title_style)
            story.append(title)
            
            # Date
            date_para = Paragraph(f"<b>Date:</b> {date_str}", styles['Normal'])
            story.append(date_para)
            story.append(Spacer(1, 20))
            
            # Summary statistics
            summary_title = Paragraph("<b>Summary Statistics</b>", styles['Heading2'])
            story.append(summary_title)
            story.append(Spacer(1, 10))
            
            summary_data = [
                ['Metric', 'Value'],
                ['Total Violations', str(stats.get('total_violations', 0))],
                ['Helmet Compliant', str(stats.get('helmet_count', 0))],
                ['No Helmet', str(stats.get('no_helmet_count', 0))],
                ['Compliance Rate', f"{stats.get('compliance_rate', 0) * 100:.2f}%"]
            ]
            
            summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 30))
            
            # Violations detail
            if violations:
                violations_title = Paragraph("<b>Violations Detail</b>", styles['Heading2'])
                story.append(violations_title)
                story.append(Spacer(1, 10))
                
                # Limit to first 50 violations for PDF
                display_violations = violations[:50]
                
                violations_data = [['Time', 'Camera', 'Type', 'Confidence', 'Severity']]
                
                for v in display_violations:
                    timestamp = v.get('timestamp', datetime.now())
                    if isinstance(timestamp, str):
                        time_str = timestamp
                    else:
                        time_str = timestamp.strftime('%H:%M:%S')
                    
                    violations_data.append([
                        time_str,
                        v.get('camera_id', 'N/A')[:10],
                        v.get('type', 'N/A'),
                        f"{v.get('confidence', 0):.2f}",
                        v.get('severity', 'N/A')
                    ])
                
                violations_table = Table(violations_data, colWidths=[1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
                violations_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(violations_table)
                
                if len(violations) > 50:
                    note = Paragraph(f"<i>Note: Showing first 50 of {len(violations)} violations</i>", styles['Normal'])
                    story.append(Spacer(1, 10))
                    story.append(note)
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF report generated: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return ""
    
    def generate_weekly_report(
        self,
        start_date: datetime,
        end_date: datetime,
        daily_stats: List[Dict]
    ) -> str:
        """
        Generate weekly summary report
        
        Args:
            start_date: Week start date
            end_date: Week end date
            daily_stats: List of daily statistics
            
        Returns:
            Path to report file
        """
        week_str = f"{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}"
        filename = f"weekly_report_{week_str}.pdf"
        filepath = self.output_dir / filename
        
        try:
            # Create visualization
            chart_path = self._create_weekly_chart(daily_stats, week_str)
            
            doc = SimpleDocTemplate(str(filepath), pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title = Paragraph("Weekly Safety Compliance Report", styles['Title'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Period
            period = Paragraph(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}", styles['Normal'])
            story.append(period)
            story.append(Spacer(1, 20))
            
            # Add chart if created
            if chart_path and Path(chart_path).exists():
                img = Image(chart_path, width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 20))
            
            # Weekly summary table
            if daily_stats:
                summary_title = Paragraph("<b>Daily Breakdown</b>", styles['Heading2'])
                story.append(summary_title)
                story.append(Spacer(1, 10))
                
                table_data = [['Date', 'Violations', 'Compliant', 'Non-Compliant', 'Rate']]
                
                for stat in daily_stats:
                    table_data.append([
                        stat.get('date', 'N/A'),
                        str(stat.get('total_violations', 0)),
                        str(stat.get('helmet_compliant', 0)),
                        str(stat.get('no_helmet', 0)),
                        f"{stat.get('compliance_rate', 0) * 100:.1f}%"
                    ])
                
                table = Table(table_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.4*inch, 1*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(table)
            
            doc.build(story)
            
            logger.info(f"Weekly report generated: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")
            return ""
    
    def _create_weekly_chart(self, daily_stats: List[Dict], week_str: str) -> Optional[str]:
        """
        Create visualization chart for weekly data
        
        Args:
            daily_stats: List of daily statistics
            week_str: Week identifier string
            
        Returns:
            Path to chart image
        """
        try:
            if not daily_stats:
                return None
            
            dates = [stat.get('date', '') for stat in daily_stats]
            compliance_rates = [stat.get('compliance_rate', 0) * 100 for stat in daily_stats]
            violations = [stat.get('total_violations', 0) for stat in daily_stats]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Compliance rate chart
            ax1.plot(dates, compliance_rates, marker='o', linewidth=2, color='#2E86AB')
            ax1.fill_between(dates, compliance_rates, alpha=0.3, color='#2E86AB')
            ax1.set_title('Daily Compliance Rate', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Compliance Rate (%)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 105])
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Violations chart
            ax2.bar(dates, violations, color='#A23B72', alpha=0.7)
            ax2.set_title('Daily Violations', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Number of Violations', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3, axis='y')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            
            chart_path = self.output_dir / f"weekly_chart_{week_str}.png"
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"Error creating weekly chart: {e}")
            return None